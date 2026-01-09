from collections.abc import Iterable
import torch
import torch.nn.functional as F
import math
from torch import nn
from itertools import islice

from transformers.configuration_utils import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from .interfaces import SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from .bailing_moe import BailingMLP, BailingMoE
from .deepseek_v2 import DeepseekV2MLAAttention

from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla.ops.simple_gla.chunk import chunk_simple_gla


def is_mla_layer(layer_idx: int, config: PretrainedConfig) -> bool:
    return (
        (layer_idx + 1) % config.layer_group_size == 0
        or layer_idx
        >= config.num_hidden_layers // config.layer_group_size * config.layer_group_size
    )


class BailingMoeV2_5GroupRMSNorm(nn.Module):
    def __init__(self, hidden_size, group_norm_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_norm_size = group_norm_size
        assert (
            hidden_size % group_norm_size == 0
        ), "hidden_size must be divisible by group_norm_size"
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.size()
        group_input_shape = input_shape[:-1] + (
            self.group_norm_size,
            input_shape[-1] // self.group_norm_size,
        )
        hidden_states = hidden_states.view(group_input_shape)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.view(input_shape).to(input_dtype)


class BailingMoeV2_5LinearAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        layer_idx = int(prefix.split(".")[-2])
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        tp_size = get_tensor_model_parallel_world_size()

        assert self.total_num_heads % tp_size == 0
        assert self.total_num_heads >= self.total_kv_heads

        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size_per_rank = self.head_dim * self.num_heads
        self.num_kv_heads = max(1, self.total_kv_heads // tp_size)
        self.kv_size_per_rank = self.num_kv_heads * self.head_dim

        self.use_qk_norm = getattr(config, "use_qk_norm", False)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=f"{prefix}.query_key_value",
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.dense",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            # rotary embedding parameters from config, if vllm version is earlier, pass rotary_dim and base instead
            # rotary_dim=config.rotary_dim,
            # base=config.rope_theta,
            rope_parameters=(
                config.rope_parameters if hasattr(config, "rope_parameters") else None
            ),
            is_neox_style=True,
        )

        self.g_norm = BailingMoeV2_5GroupRMSNorm(
            self.num_heads * self.head_dim,
            config.group_norm_size,
            eps=config.rms_norm_eps,
        )
        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )

        slope = -BailingMoeV2_5LinearAttention.build_slope_tensor(self.num_heads) * (
            1 - (layer_idx - 1) / (config.num_hidden_layers - 1) + 1e-5
        )
        self.register_buffer("slope", slope, persistent=False)

        self.lightning_attn_ops = {
            "chunk": chunk_simple_gla,
            "fused_recurrent": fused_recurrent_simple_gla,
        }

    @staticmethod
    def build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float)
        return slopes

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: (tokens, hidden_size)
        # hidden_states: (4096, 2048)
        bsz, seq_len = hidden_states.size()
        mode = "fused_recurrent" if seq_len == 1 else "chunk"

        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split(
            [self.q_size_per_rank, self.kv_size_per_rank, self.kv_size_per_rank], dim=-1
        )

        if self.use_qk_norm:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
            q = q.view(-1, self.q_size_per_rank)
            k = k.view(-1, self.kv_size_per_rank)

        q, k = self.rotary_emb(position_ids, q, k)
        # q.shape=k.shape=v.shape=(4096, 2048)
        o, _ = self.lightning_attn_ops[mode](
            q=q,
            k=k,
            v=v,
            g=self.slope[None, None, :].expand(bsz, seq_len, self.num_heads),
            initial_state=None,
            output_final_state=False,
        )
        o = o.reshape(bsz, seq_len, -1)
        o = self.g_norm(o)
        g_proj = self.g_proj(hidden_states)
        o = o * torch.sigmoid_(g_proj)
        o = self.dense(o)

        return o


class BailingMoeV2_5Block(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        layer_idx = int(prefix.split(".")[-1])
        self.layer_idx = layer_idx
        self.config = config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        if self._is_mla_layer():
            attn_cls = DeepseekV2MLAAttention
            attn_args = {
                "vllm_config": vllm_config,
                "config": config,
                "hidden_size": hidden_size,
                "num_heads": config.num_attention_heads,
                "qk_nope_head_dim": config.qk_nope_head_dim,
                "qk_rope_head_dim": config.qk_rope_head_dim,
                "v_head_dim": config.v_head_dim,
                "q_lora_rank": config.q_lora_rank,
                "kv_lora_rank": config.kv_lora_rank,
                "prefix": f"{prefix}.attention",
            }
        else:
            attn_cls = BailingMoeV2_5LinearAttention
            attn_args = {
                "config": config,
                "cache_config": cache_config,
                "quant_config": quant_config,
                "prefix": f"{prefix}.attention",
            }

        self.attention = attn_cls(**attn_args)

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        # Choose MLP class based on the number of experts and layer index
        if layer_idx < config.first_k_dense_replace:
            mlp_class = BailingMLP
        else:
            mlp_class = BailingMoE
        self.mlp = mlp_class(
            intermediate_size, config, quant_config, True, prefix=f"{prefix}.mlp"
        )

    def _is_mla_layer(self) -> bool:
        return is_mla_layer(self.layer_idx, self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            print(f"Loading weight: {name} into {self.layer_idx} {loaded_weight.shape}")
        return loaded_params


@support_torch_compile
class BailingMoeV2_5Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        if get_pp_group().is_first_rank or (
            self.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.word_embeddings",
            )
        else:
            self.word_embeddings = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: BailingMoeV2_5Block(
                vllm_config=vllm_config,
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                hidden_states,
                position_ids,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        else:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        # name: ckpt weight name
        # loaded_weight: ckpt weight tensor
        # param_dict: model parameter dict
        for name, loaded_weight in weights:
            if (
                hasattr(self.config, "norm_head")
                and self.config.norm_head
                and "lm_head.weight" in name
            ):
                loaded_weight = F.normalize(loaded_weight, dim=0, p=2, eps=1e-7)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # replace mla dense param with o_proj
                    if "dense" in name:
                        layer_idx = int(name.split("layers.")[1].split(".")[0])
                        if is_mla_layer(layer_idx, self.config):
                            name = name.replace("dense", "o_proj")

                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class BailingMoeV2_5ForCausalLM(nn.Module, SupportsPP):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        vllm_config.model_config.hf_config = config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.max_position_embeddings = config.max_position_embeddings
        self.model = BailingMoeV2_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        if get_pp_group().is_last_rank:
            if self.tie_word_embeddings:
                self.lm_head = self.model.word_embeddings
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
