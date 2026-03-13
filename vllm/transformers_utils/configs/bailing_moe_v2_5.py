# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BailingMoeV2_5Config(PretrainedConfig):
    model_type = "bailing_moe_linear"

    def __init__(
        self,
        attention_dropout=0.0,
        first_k_dense_replace=1,
        group_norm_size=4,
        head_dim=128,
        hidden_act="silu",
        hidden_size=2048,
        intermediate_size=5120,
        kv_lora_rank=512,
        layer_group_size=5,
        max_position_embeddings=32768,
        moe_intermediate_size=512,
        router_dtype=None,
        moe_shared_expert_intermediate_size=512,
        n_group=8,
        n_routed_experts=256,
        norm_topk_prob=True,
        num_attention_heads=16,
        num_experts=256,
        num_experts_per_tok=8,
        num_hidden_layers=20,
        num_key_value_heads=16,
        num_kv_heads_for_linear_attn=16,
        num_nextn_predict_layers=1,
        num_shared_experts=1,
        q_lora_rank=None,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        routed_scaling_factor=2.5,
        score_function="sigmoid",
        tie_word_embeddings=False,
        layer_types=None,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.attention_dropout = attention_dropout
        self.first_k_dense_replace = first_k_dense_replace
        self.group_norm_size = group_norm_size
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.kv_lora_rank = kv_lora_rank
        self.layer_group_size = layer_group_size
        self.max_position_embeddings = max_position_embeddings
        self.moe_intermediate_size = moe_intermediate_size
        self.router_dtype = router_dtype
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.n_group = n_group
        self.n_routed_experts = n_routed_experts
        self.norm_topk_prob = norm_topk_prob
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_kv_heads_for_linear_attn = num_kv_heads_for_linear_attn
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_shared_experts = num_shared_experts
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.routed_scaling_factor = routed_scaling_factor
        self.score_function = score_function
        self.tie_word_embeddings = tie_word_embeddings

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if bool((i + 1) % self.layer_group_size)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        layer_type_validation(self.layer_types)
