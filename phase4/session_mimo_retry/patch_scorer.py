src = open('probe_eot2.py', encoding='utf-8').read()
start = src.index('# ===== scorer copied from eval_pass.py =====')
end = src.index('# ===== end scorer =====')
new_block = '''# ===== scorer extracted verbatim from eval_pass.py =====
_ev = open("eval_pass.py", encoding="utf-8").read()
_a = _ev.index("    import re")
_b = _ev.index("    n, capped, closed, correct")
_exec_src = _ev[_a:_b]
_exec_src = chr(10).join(l[4:] if l.startswith("    ") else l for l in _exec_src.splitlines())
exec(_exec_src, globals())
'''
src = src[:start] + new_block + src[end:]
open('probe_eot2.py', 'w', encoding='utf-8').write(src)
print('patched')
