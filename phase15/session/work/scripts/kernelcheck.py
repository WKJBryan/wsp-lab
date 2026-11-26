"""Do vLLM's fast kernels match the transformers reference implementations numerically?

Compares chunk_gated_delta_rule and causal_conv1d (vLLM triton) with the reference PyTorch
functions transformers falls back to, on random inputs of the model's shapes, for several lengths.
"""
import torch, inspect, time
torch.set_grad_enabled(False)
import transformers.models.qwen3_5.modeling_qwen3_5 as Q
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule as fast_chunk
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn as fast_conv
print('fast chunk:', inspect.signature(fast_chunk))
print('ref  chunk:', inspect.signature(Q.torch_chunk_gated_delta_rule))
print('fast conv :', inspect.signature(fast_conv))
dev = 'cuda'
H, DK, DV = 16, 128, 128
for T in [200, 1536, 4096]:
    q = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, device=dev, dtype=torch.float32) * 0.3
    beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
    kw = dict(initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
    torch.cuda.synchronize(); t = time.time()
    o_ref = Q.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, **kw)
    torch.cuda.synchronize(); t_ref = time.time() - t
    t = time.time()
    o_fast = fast_chunk(q, k, v, g=g, beta=beta, **kw)
    torch.cuda.synchronize(); t_fast = time.time() - t
    if isinstance(o_ref, tuple): o_ref = o_ref[0]
    if isinstance(o_fast, tuple): o_fast = o_fast[0]
    d = (o_ref.float() - o_fast.float()).abs()
    print(f'T={T}: GDN ref {t_ref:.2f}s fast {t_fast:.2f}s | out shape {tuple(o_fast.shape)} ref {tuple(o_ref.shape)} | max abs diff {float(d.max()):.4f} mean {float(d.mean()):.5f} | out rms {float(o_ref.float().pow(2).mean().sqrt()):.3f}')
C = 6144
for T in [200, 4096]:
    x = torch.randn(1, C, T, device=dev, dtype=torch.bfloat16)
    w = torch.randn(C, 4, device=dev, dtype=torch.bfloat16) * 0.1
    b = torch.randn(C, device=dev, dtype=torch.bfloat16)
    ref_conv = Q.causal_conv1d_fn
    torch.cuda.synchronize(); t = time.time()
    y_ref = ref_conv(x, w, b, activation='silu')
    torch.cuda.synchronize(); t_ref = time.time() - t
    t = time.time()
    y_fast = fast_conv(x, w, b, activation='silu')
    torch.cuda.synchronize(); t_fast = time.time() - t
    d = (y_ref.float() - y_fast.float()).abs()
    print(f'T={T}: conv ref {t_ref:.2f}s fast {t_fast:.2f}s | shapes {tuple(y_ref.shape)} {tuple(y_fast.shape)} | max abs diff {float(d.max()):.4f} mean {float(d.mean()):.5f}')
