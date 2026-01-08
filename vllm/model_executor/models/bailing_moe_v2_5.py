import torch
import torch.nn.functional as F
from torch import nn
from itertools import islice

from transformers.configuration_utils import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.sequence import IntermediateTensors
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
    QKVParallelLinear,
    RowParallelLinear,
)
from .bailing_moe import BailingMLP, BailingMoE
from .deepseek_v2 import DeepseekV2MLAAttention

from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla.ops.simple_gla.chunk import chunk_simple_gla


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
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
        )

        self.lightning_attn_ops = {
            "chunk": chunk_simple_gla,
            "fused_recurrent": fused_recurrent_simple_gla,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        mode = "fused_recurrent" if hidden_states.size(1) == 1 else "chunk"

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


class BailingMoeV2_5Block(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        layer_idx = int(prefix.split(".")[-1])
        self.config = config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        if (
            (layer_idx + 1) % config.layer_group_size == 0
            or layer_idx >= config.num_hidden_layers // config.layer_group_size**2
        ):
            attn_cls = DeepseekV2MLAAttention
        else:
            attn_cls = BailingMoeV2_5LinearAttention

        self.attention = attn_cls(
            config, cache_config, quant_config, prefix=f"{prefix}.attention"
        )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        # Choose MLP class based on the number of experts and layer index
        if layer_idx < config.first_k_dense_replace:
            mlp_class = BailingMLP
        else:
            mlp_class = BailingMoE
        self.mlp = mlp_class(
            intermediate_size, config, quant_config, True, prefix=f"{prefix}.mlp"
        )

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
