"""Filter rows of a val_E2 output: never-closed traces (or others) and where the stop neuron would first close them.
usage: python filt.py <file> [closed 0/1/-]"""
import sys

f = sys.argv[1]
cl = sys.argv[2] if len(sys.argv) > 2 else "0"
lines = open(f, encoding="utf-8").read().splitlines()
for i, line in enumerate(lines):
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 7 or not parts[0][:1] == "p" or not parts[0][1:].isdigit():
        continue
    c, o, lv = parts[1].split()
    if cl != "-" and c != cl:
        continue
    nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
    if parts[3] == "None":
        continue
    print(f"{parts[0]:5s} cl/ok {c}{o} {lv} | first close row,P,top: {parts[3]} | stated: {parts[4][:36]} | {nxt[:170]}")
