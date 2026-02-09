# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import islice

import torch
from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.bailing_moe import BailingMLP, BailingMoE
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
from vllm.model_executor.models.interfaces import IsHybrid, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.gla_attn import GLAAttentionBackend

logger = init_logger(__name__)


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
        self.weight.weight_loader = self.weight_loader

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        assert self.tp_size <= group_norm_size, (
            f"Tensor model parallel size {self.tp_size} must be less than or equal to group norm size {group_norm_size}."  # noqa: E501
        )
        assert group_norm_size % self.tp_size == 0, (
            f"Group norm size {group_norm_size} must be divisible by tensor model parallel size {self.tp_size}."  # noqa: E501
        )
        assert hidden_size % group_norm_size == 0, (
            f"Hidden size {hidden_size} must be divisible by group norm size {group_norm_size}."  # noqa: E501
        )

        self.group_norm_size = group_norm_size // self.tp_size
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
        return hidden_states.view(input_shape).to(input_dtype) * self.weight

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        loaded_weight = loaded_weight.view(self.tp_size, -1)[self.tp_rank].contiguous()
        param_data.copy_(loaded_weight)


class BailingMoeV2_5LinearAttention(nn.Module, MambaBase):
    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.layer_idx = int(prefix.split(".")[-2])
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads

        self.tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % self.tp_size == 0
        assert self.total_num_heads >= self.total_kv_heads

        self.num_heads = self.total_num_heads // self.tp_size
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        # we have same q_size_per_rank and kv_size_per_rank in bailing moe v2.5
        self.q_size_per_rank = self.head_dim * self.num_heads
        self.num_kv_heads = max(1, self.total_kv_heads // self.tp_size)
        self.kv_size_per_rank = self.num_kv_heads * self.head_dim

        self.use_qk_norm = getattr(config, "use_qk_norm", False)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=vllm_config.quant_config,
            prefix=f"{self.prefix}.query_key_value",
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=vllm_config.quant_config,
            reduce_results=reduce_results,
            prefix=f"{self.prefix}.dense",
        )

        rope_parameters = (
            config.rope_parameters
            if hasattr(config, "rope_parameters")
            else {
                "rope_theta": config.rope_theta,
                "rotary_dim": config.rotary_dim,
            }
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
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
            quant_config=vllm_config.quant_config,
            prefix=f"{self.prefix}.g_proj",
        )
        slope = -BailingMoeV2_5LinearAttention.build_slope_tensor(self.num_heads) * (
            1 - self.layer_idx / (config.num_hidden_layers - 1) + 1e-5
        )
        self.register_buffer("slope", slope)

    @property
    def mamba_type(self) -> str:
        return "gla_attention"

    def get_state_shape(self):
        return MambaStateShapeCalculator.linear_attention_state_shape(
            self.total_num_heads, self.tp_size, self.head_dim
        )

    def get_state_dtype(self):
        return MambaStateDtypeCalculator.gla_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_attn_backend(self):
        return GLAAttentionBackend

    @staticmethod
    def build_slope_tensor(num_heads: int):
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

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        heads_per_partition = num_heads // tp_size
        slopes = torch.tensor(get_slopes(num_heads))[
            tp_rank * heads_per_partition : (tp_rank + 1) * heads_per_partition
        ]
        return slopes

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attention_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        recurrent_states = self.kv_cache[forward_context.virtual_engine][0]
        attn_metadata = attn_metadata[self.prefix]
        state_indices_tensor = attn_metadata.state_indices_tensor
        query_start_loc = attn_metadata.query_start_loc

        # Clear recurrent states for new sequences in the prefill phase
        if attn_metadata.num_prefills > 0:
            num_decode_tokens = attn_metadata.num_decode_tokens
            for prefill_idx in range(attn_metadata.num_prefills):
                q_start = query_start_loc[num_decode_tokens + prefill_idx]
                q_end = query_start_loc[num_decode_tokens + prefill_idx + 1]
                query_len = q_end - q_start
                context_len = (
                    attn_metadata.seq_lens[num_decode_tokens + prefill_idx] - query_len
                )
                # Only clear state at the first chunk of prefill
                if context_len == 0:
                    block_to_clear = state_indices_tensor[
                        num_decode_tokens + prefill_idx
                    ]
                    recurrent_states[block_to_clear, ...] = 0

        num_tokens, _ = hidden_states.size()

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

        q = q.view(1, num_tokens, self.num_heads, self.head_dim)
        k = k.view(1, num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(1, num_tokens, self.num_kv_heads, self.head_dim)

        recurrent_state = torch.index_select(recurrent_states, 0, state_indices_tensor)

        o, recurrent_state = fused_recurrent_simple_gla(
            q=q,
            k=k,
            v=v,
            g=self.slope[None, None, :].expand(1, num_tokens, self.num_heads),
            initial_state=recurrent_state,
            output_final_state=True,
            cu_seqlens=query_start_loc,
        )

        o = o.to(hidden_states.dtype)
        recurrent_state = recurrent_state.to(recurrent_states.dtype)
        recurrent_states.index_put_((state_indices_tensor,), recurrent_state)

        o = o.reshape(num_tokens, self.q_size_per_rank)
        o = self.g_norm(o)
        g_proj, _ = self.g_proj(hidden_states)
        o = o * torch.sigmoid(g_proj)
        o, _ = self.dense(o)

        return o


class BailingMoeV2_5Block(nn.Module):
    def __init__(
        self, vllm_config: VllmConfig, config: PretrainedConfig, prefix: str = ""
    ):
        super().__init__()
        self.layer_idx = int(prefix.split(".")[-1])
        self.config = config

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self._is_mla_layer():
            self.attention = DeepseekV2MLAAttention(
                config=config,
                vllm_config=vllm_config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.attention = BailingMoeV2_5LinearAttention(
                config=config,
                vllm_config=vllm_config,
                reduce_results=True,
                prefix=f"{prefix}.self_attn",
            )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )

        mlp_cls = (
            BailingMLP if self.layer_idx < config.first_k_dense_replace else BailingMoE
        )
        self.mlp = mlp_cls(
            intermediate_size=config.intermediate_size,
            config=config,
            quant_config=vllm_config.quant_config,
            reduce_results=True,
            prefix=f"{prefix}.mlp",
        )

    def _is_mla_layer(self) -> bool:
        return is_mla_layer(self.layer_idx, self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_kwargs = {
            "hidden_states": hidden_states,
        }
        if self._is_mla_layer():
            attn_kwargs["positions"] = position_ids
        else:
            attn_kwargs["position_ids"] = position_ids

        attn_output = self.attention(**attn_kwargs)

        if attn_output is None:
            # NOTE: when we run v1 profile run, attn_output is None
            attn_output = torch.zeros_like(hidden_states)

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@support_torch_compile
class BailingMoeV2_5Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)
        pp_group = get_pp_group()
        if pp_group.is_first_rank or (tie_word_embeddings and pp_group.is_last_rank):
            self.word_embeddings = VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.word_embeddings",
            )
        else:
            self.word_embeddings = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: BailingMoeV2_5Block(
                vllm_config=vllm_config,
                config=self.config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if pp_group.is_last_rank:
            self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        itermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert itermediate_tensors is not None
            hidden_states = itermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
            )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        else:
            hidden_states = self.norm(hidden_states)
            return hidden_states

    def get_expert_mapping(self):
        return SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights):
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params = set()
        expert_params_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            # skip missing parameters in PP and parameters that are not in the model
            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                logger.warning(f"Parameter {name} not found in the model. Skipping.")  # noqa: G004
                continue
            # process expert weights
            if "mlp.experts" in name:
                for mapping in expert_params_mapping:
                    stacked_param_name, sharded_weight_name, expert_id, shared_id = (
                        mapping
                    )
                    if sharded_weight_name not in name:
                        continue
                    name = name.replace(sharded_weight_name, stacked_param_name)

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    # vllm/model_executor/layers/fused_moe/layer.py
                    # FusedMoE.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shared_id=shared_id,
                        expert_id=expert_id,
                    )
                    break
                loaded_params.add(name)
                continue
            # process shared weights like gate_up_proj and fused_qkv_a_proj
            for (
                stacked_param_name,
                sharded_weight_name,
                shared_id,
            ) in stacked_params_mapping:
                if sharded_weight_name not in name:
                    continue
                name = name.replace(sharded_weight_name, stacked_param_name)
                # QKV fusion is optional
                if stacked_param_name == "fused_qkv_a_proj" and name not in params_dict:
                    continue

                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shared_id)
                loaded_params.add(name)
                break
            # process other weights when there is no break in the above loop
            # ßwhich means the weight is not shared
            else:
                # rename dense to o_proj for MLA layers
                if "dense" in name:
                    layer_idx = int(name.split("layers.")[1].split(".")[0])
                    if is_mla_layer(layer_idx, self.config):
                        name = name.replace("dense", "o_proj")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


class BailingMoeV2_5ForCausalLM(nn.Module, SupportsPP, IsHybrid):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)

        self.model = BailingMoeV2_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if self.tie_word_embeddings:
                self.lm_head = self.model.word_embeddings
            else:
                self.lm_head = ParallelLMHead(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    quant_config=vllm_config.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
            self.logits_processor = LogitsProcessor(self.config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> dict[str, torch.dtype]:
        return MambaStateDtypeCalculator.gla_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        return MambaStateShapeCalculator.linear_attention_state_shape(
            hf_config.num_attention_heads, tp_size, hf_config.head_dim
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights) -> set[str]:
        loader = AutoWeightsLoader(
            self, skip_prefixes=(["lm_head."] if self.tie_word_embeddings else None)
        )
        return loader.load_weights(weights)
