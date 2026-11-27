import inspect
import transformers.models.qwen3_5.modeling_qwen3_5 as m
print(inspect.getsource(m.Qwen3_5RMSNorm))
print(inspect.getsource(m.create_causal_mask)[:1500])
import transformers.masking_utils as mu
print([n for n in dir(mu) if 'mask' in n.lower()])
