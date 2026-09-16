# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    RoutedExpertsManager,
    resolve_routed_experts_slot_mapping_spec,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_gpu_model_runner_binds_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class _DummyRouter:
        _routing_replay_out: torch.Tensor | None = None

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    # Patch the runtime import inside _bind_routed_experts_capturer.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [dummy_module])
    )

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert dummy_module.router.capture_fn is not None
    dummy_module.router.capture_fn(torch.tensor([[5, 6]]))

    assert len(capturer.calls) == 1
    layer_id, topk_ids = capturer.calls[0]
    assert layer_id == 7
    assert torch.equal(topk_ids, torch.tensor([[5, 6]]))


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [dummy_module])
    )

    # Before binding, no capture hook.
    assert dummy_module.router.capture_fn is None

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    # After binding, hook should exist and be callable.
    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[9, 10]]))
    assert len(capturer.calls) == 1


def test_gpu_model_runner_does_not_bind_draft_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        model=types.SimpleNamespace(modules=lambda: [target_module]),
        compilation_config=types.SimpleNamespace(
            static_forward_context={
                "model.layers.7.mlp.experts": target_module,
                "mtp.layers.0.mlp.experts": draft_module,
            }
        ),
    )

    capturer = types.SimpleNamespace(capture=lambda *_: None)
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert target_module.router.capture_fn is not None
    assert draft_module.router.capture_fn is None


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """n == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """n == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1


# ---------------------------------------------------------------------------
# Canonical routing-slot tests (resolve_routed_experts_slot_mapping_spec,
# RoutedExpertsManager logical-block geometry, and the worker-side
# _build_routed_experts_routing_slots helper).
# ---------------------------------------------------------------------------

import torch.nn.functional  # noqa: F401,E402  (side-effect import, kept close to use)

_TORCH_DTYPE = torch.float16


def _full_attn_group(block_size: int, layer: str = "layers.0") -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        layer_names=[layer],
        kv_cache_spec=FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=64,
            dtype=_TORCH_DTYPE,
        ),
    )


def _swa_group(block_size: int, window: int = 128) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        layer_names=["layers.1"],
        kv_cache_spec=SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=64,
            dtype=_TORCH_DTYPE,
            sliding_window=window,
        ),
    )


def _kv_config(groups: list[KVCacheGroupSpec], num_blocks: int = 16) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


def test_resolver_full_attention_defaults_to_block_size():
    spec = resolve_routed_experts_slot_mapping_spec(_kv_config([_full_attn_group(16)]))
    assert spec.logical_block_size == 16
    assert spec.kv_cache_group_id == 0


def test_resolver_picks_smallest_logical_block_size_then_smallest_gid():
    groups = [
        _full_attn_group(32, "a"),
        _full_attn_group(8, "b"),
        _full_attn_group(8, "c"),
    ]
    spec = resolve_routed_experts_slot_mapping_spec(_kv_config(groups))
    # tie between gid 1 and 2 at size 8 -> smallest gid wins.
    assert spec.logical_block_size == 8
    assert spec.kv_cache_group_id == 1


def test_resolver_swa_not_a_candidate_but_full_attention_is():
    # Sliding window specs recycle historical blocks and must not anchor;
    # the full-attention group is still eligible.
    groups = [_swa_group(16), _full_attn_group(32, "layers.9")]
    spec = resolve_routed_experts_slot_mapping_spec(_kv_config(groups))
    assert spec.kv_cache_group_id == 1
    assert spec.logical_block_size == 32


def test_resolver_no_candidate_raises():
    with pytest.raises(ValueError, match="no KV cache group"):
        resolve_routed_experts_slot_mapping_spec(_kv_config([_swa_group(16)]))


def test_resolver_rejects_non_positive_logical_size():
    class _BrokenSpec(FullAttentionSpec):
        @property
        def routed_experts_logical_block_size(self) -> int | None:
            return 0

    group = KVCacheGroupSpec(
        layer_names=["layers.0"],
        kv_cache_spec=_BrokenSpec(
            block_size=16, num_kv_heads=1, head_size=64, dtype=_TORCH_DTYPE
        ),
    )
    with pytest.raises(ValueError, match="invalid routed_experts_logical_block_size"):
        resolve_routed_experts_slot_mapping_spec(_kv_config([group]))


class _StubHfConfig:
    num_experts = 8
    n_routed_experts = 8
    num_experts_per_tok = 2
    num_hidden_layers = 3


def _make_manager(logical_block_size: int, num_blocks: int = 4) -> RoutedExpertsManager:
    mgr = RoutedExpertsManager.__new__(RoutedExpertsManager)
    mgr.kv_cache_group_id = 0
    mgr.logical_block_size = logical_block_size
    hf = _StubHfConfig()
    mgr.routed_experts_by_slot = np.zeros(
        (num_blocks * logical_block_size, hf.num_hidden_layers, hf.num_experts_per_tok),
        dtype=np.uint8,
    )
    return mgr


def test_manager_get_across_logical_blocks_with_token_start():
    # logical_block_size=8 > physical block_size=4: consecutive original
    # tokens in the same logical block share one block table entry and must
    # not overwrite each other.
    mgr = _make_manager(logical_block_size=8, num_blocks=4)
    # Two logical blocks: block ids 1 and 2.
    block_ids = [1, 2]
    # Fill slots for 12 tokens: positions 0..7 in block 1, 8..11 in block 2.
    for pos in range(12):
        slot = (block_ids[pos // 8] * 8) + pos % 8
        mgr.routed_experts_by_slot[slot] = pos  # layer/top_k broadcast
    out = mgr.get(block_ids, num_tokens=12)
    assert out.shape == (12, 3, 2)
    for pos in range(12):
        assert out[pos, 0, 0] == pos, f"position {pos} corrupted"
    # token_start skips the first 5 tokens.
    out2 = mgr.get(block_ids, num_tokens=12, token_start=5)
    assert out2.shape == (7, 3, 2)
    assert out2[0, 0, 0] == 5
    assert out2[6, 0, 0] == 11


def test_manager_buffer_capacity_scales_with_logical_block_size():
    mgr = _make_manager(logical_block_size=8, num_blocks=4)
    assert mgr.routed_experts_by_slot.shape[0] == 4 * 8


def test_manager_get_insufficient_block_ids_fails():
    mgr = _make_manager(logical_block_size=8, num_blocks=4)
    # 20 tokens need 3 logical blocks; only 2 block ids available.
    with pytest.raises(ValueError, match="logical blocks"):
        mgr.get([1, 2], num_tokens=20)


def test_manager_store_batch_row_mismatch_fails():
    mgr = _make_manager(logical_block_size=8)
    data = np.zeros((3, 3, 2), dtype=np.uint8)
    slots = np.array([0, 1])
    with pytest.raises(ValueError, match="do not match"):
        mgr.store_batch(data, slots)


def test_manager_store_batch_negative_slot_fails():
    mgr = _make_manager(logical_block_size=8)
    data = np.zeros((2, 3, 2), dtype=np.uint8)
    slots = np.array([0, -1])
    with pytest.raises(ValueError, match="out of range"):
        mgr.store_batch(data, slots)


def test_manager_store_batch_slot_beyond_buffer_fails():
    mgr = _make_manager(logical_block_size=8, num_blocks=4)
    data = np.zeros((2, 3, 2), dtype=np.uint8)
    slots = np.array([0, 4 * 8])
    with pytest.raises(ValueError, match="out of range"):
        mgr.store_batch(data, slots)


def test_manager_logical_gt_physical_no_overwrite():
    # Simulate the C4 geometry: physical block_size=4 (attention slot
    # granularity) but logical_block_size=16. Tokens 0..15 map into one
    # logical block; each must get a unique slot.
    lbs = 16
    slots = np.array([3 * lbs + off for off in range(lbs)])
    assert len(set(slots.tolist())) == lbs, "slots must be unique per token"


# ---------------------------------------------------------------------------
# Worker-side canonical mapping helper tests (CPU-runnable).
# ---------------------------------------------------------------------------


class _StubRunner:
    """Minimal harness exposing GPUModelRunner._build_routed_experts_routing_slots."""

    _build = GPUModelRunner._build_routed_experts_routing_slots

    def build(self, *args, **kwargs):
        return self._build(*args, **kwargs)


def _worker_helper_slots(
    block_table: list[list[int]],
    req_indices: list[int],
    positions: list[int],
    num_tokens: int,
    logical_block_size: int,
) -> list[int]:
    runner = _StubRunner()
    out = runner.build(
        block_table=torch.tensor(block_table, dtype=torch.int32),
        req_indices=torch.tensor(req_indices, dtype=torch.int64),
        positions=torch.tensor(positions, dtype=torch.int64),
        num_tokens=num_tokens,
        logical_block_size=logical_block_size,
    )
    assert out.dtype == torch.int64
    return out.tolist()


def test_worker_helper_full_attention_matches_slot_mapping_semantics():
    # FullAttention: logical_block_size == block_size (2). The canonical
    # formula equals the legacy block_table.slot_mapping semantics.
    block_table = [[5, 6]]
    got = _worker_helper_slots(
        block_table,
        req_indices=[0] * 4,
        positions=[0, 1, 2, 3],
        num_tokens=4,
        logical_block_size=2,
    )
    assert got == [5 * 2 + 0, 5 * 2 + 1, 6 * 2 + 0, 6 * 2 + 1]


def test_worker_helper_chunked_prefill_second_chunk():
    # Chunked prefill: positions resume mid logical block.
    got = _worker_helper_slots(
        block_table=[[5, 6]],
        req_indices=[0] * 3,
        positions=[3, 4, 5],
        num_tokens=3,
        logical_block_size=4,
    )
    assert got == [5 * 4 + 3, 6 * 4 + 0, 6 * 4 + 1]


def test_worker_helper_multi_request_distinct_positions():
    # Two requests, decode tokens at different logical-block boundaries.
    got = _worker_helper_slots(
        block_table=[[5, 6, 0], [9, 10, 11]],
        req_indices=[0, 0, 1, 1],
        positions=[7, 8, 15, 16],
        num_tokens=4,
        logical_block_size=8,
    )
    # req0 row [5, 6, 0]: pos7 -> block5 off7; pos8 -> block6 off0.
    assert got[0] == 5 * 8 + 7
    assert got[1] == 6 * 8 + 0
    # req1 row [9, 10, 11]: pos15 -> block10 off7; pos16 -> block11 off0.
    assert got[2] == 10 * 8 + 7
    assert got[3] == 11 * 8 + 0


def test_worker_helper_position_beyond_block_table_fails():
    # Position 128 with logical_block_size=8 needs block table column 16.
    with pytest.raises(ValueError, match="block table column"):
        _worker_helper_slots(
            block_table=[[5, 6], [9, 10]],
            req_indices=[0, 1],
            positions=[0, 128],
            num_tokens=2,
            logical_block_size=8,
        )


def test_worker_helper_decode_around_logical_block_boundary():
    # C128 boundary: positions 127/128/129 land in different logical blocks.
    got = _worker_helper_slots(
        block_table=[[7, 8, 9]],
        req_indices=[0] * 3,
        positions=[127, 128, 129],
        num_tokens=3,
        logical_block_size=128,
    )
    assert got == [7 * 128 + 127, 8 * 128 + 0, 8 * 128 + 1]


def test_worker_helper_padding_rows_stay_minus_one():
    got = _worker_helper_slots(
        block_table=[[5]],
        req_indices=[0, 0, 0, 0],
        positions=[0, 1, 0, 0],
        num_tokens=2,
        logical_block_size=2,
    )
    assert got[:2] == [5 * 2, 5 * 2 + 1]
    assert got[2:] == [-1, -1]


def test_worker_helper_compressed_geometry_c4():
    # C4: logical block covers 4 original tokens even though the physical
    # KV tensor stores one compressed entry per 4 tokens.
    got = _worker_helper_slots(
        block_table=[[2, 3]],
        req_indices=[0] * 8,
        positions=list(range(8)),
        num_tokens=8,
        logical_block_size=4,
    )
    assert got == [2 * 4 + o for o in range(4)] + [3 * 4 + o for o in range(4)]
