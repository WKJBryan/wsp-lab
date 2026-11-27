import inspect
import transformers.models.qwen3_5.modeling_qwen3_5 as m
for fn in [m.torch_chunk_gated_delta_rule, m.torch_recurrent_gated_delta_rule, m.causal_conv1d_fn, m.use_kernelized_func]:
    print('=====', fn.__name__, inspect.getsourcefile(fn))
    try:
        print(inspect.getsource(fn)[:9000])
    except Exception as e:
        print('ERR', e)
print('ALL_ATTENTION_FUNCTIONS in module:', m.ALL_ATTENTION_FUNCTIONS, m.ALL_ATTENTION_FUNCTIONS is not None)
print(inspect.getsource(m.apply_rotary_pos_emb))
