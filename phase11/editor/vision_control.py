"""Free: vision tower as CONTROL (irrelevant to text eval → presumably undamaged).
Compare norm-weight 'pile at max' signatures between language and vision towers.
Also dump per-task long-text logprobs and test v_proj permutation.
"""

import json
import struct
from pathlib import Path

MODEL = Path("/home/user/wsp11/sandbox/model/model.safetensors")


def main():
    import torch
    from safetensors.torch import safe_open

    with open(MODEL, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)

    def pile_stats(t):
        """fraction at max, fraction at min, n_unique"""
        flat = t.float().flatten()
        mx = flat.max()
        mn = flat.min()
        at_max = (flat == mx).float().mean().item()
        at_min = (flat == mn).float().mean().item()
        return at_max, at_min, flat.unique().numel(), flat.numel()

    print("=== LANGUAGE tower norm piles ===")
    for name in sorted(hdr):
        if not name.startswith("model.language_model.layers"):
            continue
        if not any(k in name for k in ("layernorm", "q_norm", "k_norm")):
            continue
        if "layernorm" not in name:
            continue
        t = f.get_tensor(name) if False else None
    with safe_open(str(MODEL), framework="pt") as f:
        for name in sorted(hdr):
            if ("model.language_model" in name and "layernorm" in name
                    and "layers" in name):
                t = f.get_tensor(name)
                am, an, u, n = pile_stats(t)
                if am > 0.10:
                    print(f"  {name.split('layers.')[1]:55s} "
                          f"at_max={am:.1%} at_min={an:.1%} uniq={u}/{n}")

        print("\n=== VISION tower norm piles (control) ===")
        for name in sorted(hdr):
            if ("model.vision_tower" in name and "layernorm" in name):
                t = f.get_tensor(name)
                am, an, u, n = pile_stats(t)
                if am > 0.10:
                    print(f"  {name.split('encoder.')[1]:55s} "
                          f"at_max={am:.1%} at_min={an:.1%} uniq={u}/{n}")

        print("\n=== VISION qk-norm (control for constant-qk hypothesis) ===")
        for name in sorted(hdr):
            if ("model.vision_tower" in name
                    and ("q_norm" in name or "k_norm" in name)
                    and "layers.0." in name or "layers.13." in name):
                t = f.get_tensor(name).float()
                print(f"  {name.split('encoder.')[1]:55s} "
                      f"std={t.std():.4g} uniq={t.unique().numel()}")

        print("\n=== norm pile rule hunt: pile value vs same-layer stats ===")
        # For each language layer, get post_ff pile value (=max) and compare to
        # stats of other tensors in that layer
        import collections
        layer_tensors = collections.defaultdict(dict)
        for name in hdr:
            if "model.language_model.layers" in name:
                parts = name.split(".")
                li = int(parts[3])
                layer_tensors[li][name.split(f"layers.{li}.")[1]] = name

        for li in [0, 1, 5, 10, 11, 20, 29]:
            row = []
            for suffix in ("post_feedforward_layernorm.weight",
                           "post_attention_layernorm.weight",
                           "input_layernorm.weight",
                           "pre_feedforward_layernorm.weight",
                           "pre_feedforward_layernorm_2.weight",
                           "post_feedforward_layernorm_1.weight",
                           "post_feedforward_layernorm_2.weight"):
                nm = layer_tensors[li].get(suffix)
                if nm is None:
                    continue
                t = f.get_tensor(nm).float().flatten()
                row.append(f"{suffix.split('.')[0][:12]}: "
                           f"max={t.max():.4g} mean={t.mean():.4g} "
                           f"rms={t.pow(2).mean().sqrt():.4g} "
                           f"med={t.median():.4g}")
            print(f"  L{li}:")
            for r in row:
                print(f"     {r}")


if __name__ == "__main__":
    main()
