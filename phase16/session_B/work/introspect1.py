import inspect
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import transformers.models.qwen3_5.modeling_qwen3_5 as m
print(list(ALL_ATTENTION_FUNCTIONS.valid_keys()) if hasattr(ALL_ATTENTION_FUNCTIONS, 'valid_keys') else dir(ALL_ATTENTION_FUNCTIONS))
print(inspect.getsource(ALL_ATTENTION_FUNCTIONS.sdpa) if hasattr(ALL_ATTENTION_FUNCTIONS, 'sdpa') else inspect.getsource(type(ALL_ATTENTION_FUNCTIONS)))
print(inspect.getsource(m.eager_attention_forward))
print([n for n in dir(m) if 'rope' in n.lower() or 'rotary' in n.lower()])
print(inspect.getsource(m.Qwen3_5TextRotaryEmbedding))
import transformers.ops as ops
print([n for n in dir(ops) if 'delta' in n.lower() or 'conv' in n.lower()])
src = inspect.getsource(ops.chunk_gated_delta_rule)
print(src[:7000])
