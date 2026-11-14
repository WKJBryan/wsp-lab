import json
d=json.load(open("/home/bladless/wsp4/sandbox/practice.json"))
print(type(d).__name__, len(d))
print(json.dumps(d[0], indent=2)[:600])
# check keys
ks=set()
for x in d: ks.update(x.keys())
print("keys", ks)
