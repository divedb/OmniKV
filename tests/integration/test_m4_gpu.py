"""T10 GPU continuation checks for padded latent/opaque state and HMA aliasing."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host required")

from omnikv.adapter.vllm import VllmLayoutAdapter
from omnikv.store.cpu import CpuPageStore
from omnikv.transfer.engine import PageCopyOp, RegionTransferEngine


def make_group(kind, name, *, padding=0, compressed=1, role=None, raw=None):
    # Logical latent states: one head, 16 tokens, 16 bf16 channels.
    states, content = 16 // compressed, 32
    page = states * content + padding
    spec = type(kind, (), {})()
    spec.__dict__.update(block_size=16, num_kv_heads=1, num_heads=1, head_size=16,
                         head_size_v=0, dtype=torch.bfloat16, tokens_per_state=compressed,
                         state_content_size_bytes=content, page_size_bytes=page,
                         kv_quant_mode=SimpleNamespace(name="NONE"))
    if raw is None:
        raw = torch.empty(page * 4, dtype=torch.uint8, device="cuda")
    view = raw.view(torch.bfloat16).as_strided((4, 1, states, 16),
                                              (page // 2, 16, 16, 1))
    group = SimpleNamespace(layer_names=[name], kv_cache_spec=spec, role=role)
    tensor = SimpleNamespace(layers=[name], size=raw.numel(), layer_stride=page * 4,
                             block_stride=page, offset=0)
    return raw, view, group, tensor


def copy_page(engine, placement, store, slot, block, direction):
    offset = 0
    ops = []
    for ptr, size in placement.gpu_ops(block):
        host = store.page_ptr(slot) + offset
        src, dst = (ptr, host) if direction == "d2h" else (host, ptr)
        ops.append(PageCopyOp(src, dst, size))
        offset += size
    engine.submit(direction, ops, fence=True).wait()


@pytest.mark.parametrize("padding,compressed,indexer", [(0, 1, False), (24, 1, False),
                                                       (24, 4, False), (0, 4, True)])
def test_latent_snapshot_resume_hidden_states(padding, compressed, indexer):
    torch.manual_seed(123)
    role = SimpleNamespace(name="HISPARSE_INDEXER") if indexer else None
    raw, view, group, tensor = make_group("MLAAttentionSpec", "latent", padding=padding,
                                         compressed=compressed, role=role)
    raw.fill_(0xA5)  # include alignment padding in the bitwise gate
    view[1].normal_()
    cfg = SimpleNamespace(num_blocks=4, kv_cache_groups=[group],
                          kv_cache_tensors=[tensor], kv_cache_layout="LBNHC")
    adapter = VllmLayoutAdapter()
    (desc,) = adapter.describe({"latent": view}, cfg)
    store = CpuPageStore(desc, 2)
    slot = store.acquire_page()
    engine = RegionTransferEngine()
    copy_page(engine, adapter.placements[0], store, slot, 1, "d2h")
    saved = raw[tensor.block_stride:2*tensor.block_stride].clone()
    # Actual continuation consumes latent states through K/V projections,
    # then advances autoregressively; page equality alone is not the test.
    wk = torch.randn(16, 16, device="cuda")
    wv = torch.randn(16, 16, device="cuda")
    query = torch.randn(16, device="cuda")
    def continue_from(latent):
        states = latent.float().reshape(-1, 16)
        hidden = query.clone()
        outputs = []
        for _ in range(4):
            hidden = torch.softmax((states @ wk) @ hidden / 4, dim=0) @ (states @ wv)
            outputs.append(hidden.clone())
            states = torch.cat((states, hidden[None]), dim=0)
        return torch.stack(outputs)
    baseline = continue_from(view[1])
    fresh_raw = torch.full_like(raw, 0xFF)
    _, fresh_view, _, _ = make_group("MLAAttentionSpec", "latent", padding=padding,
                                    compressed=compressed, raw=fresh_raw)
    fresh_adapter = VllmLayoutAdapter()
    fresh_adapter.describe({"latent": fresh_view}, cfg)
    copy_page(engine, fresh_adapter.placements[0], store, slot, 1, "h2d")
    assert torch.equal(fresh_raw[tensor.block_stride:2*tensor.block_stride], saved)
    assert torch.equal(continue_from(fresh_view[1]), baseline)
    assert torch.all(fresh_raw[:tensor.block_stride] == 0xFF)
    store.release(slot)


def test_hma_overlaid_pool_preserves_other_group_owned_blocks():
    raw, a, ga, ta = make_group("MLAAttentionSpec", "a")
    _, b, gb, tb = make_group("MLAAttentionSpec", "b", raw=raw)
    a[1].fill_(3)
    b[2].fill_(7)
    cfg = SimpleNamespace(num_blocks=4, kv_cache_groups=[ga, gb],
                          kv_cache_tensors=[ta, tb], kv_cache_layout="LBNHC")
    adapter = VllmLayoutAdapter()
    descs = adapter.describe({"a": a, "b": b}, cfg)
    store = CpuPageStore(descs[0], 1)
    slot = store.acquire_page()
    engine = RegionTransferEngine()
    copy_page(engine, adapter.placements[0], store, slot, 1, "d2h")
    a[1].fill_(float("nan"))
    copy_page(engine, adapter.placements[0], store, slot, 1, "h2d")
    assert torch.all(a[1] == 3)
    assert torch.all(b[2] == 7)
    store.release(slot)
