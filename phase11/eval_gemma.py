"""Phase 11 evaluator: MATH-500 on Gemma-4 arms, vLLM, run INSIDE WSL2.

Decoding = Gemma-4 model card for every arm (generation_config.json):
temperature 1.0, top_p 0.95, top_k 64, 1 sample, cap 32768, fixed seed.
Scoring reuses the phase-3c rescore extraction (boxed first, then answer-
marker lines, then last line) with the </think> gate removed — Gemma has no
think blocks; the change is identical across every arm.

Usage:
  ~/wsp3c-venv/bin/python eval_gemma.py --model ~/wsp11/gemma4-8exp \
      --name a1_wound_practice --n 64
  ~/wsp3c-venv/bin/python eval_gemma.py --model ~/wsp11/gemma4-resharded \
      --name a0_full --n 500 --fp8
"""

import argparse
import json
import sys
from pathlib import Path

CAP = 32768
SEED = 20261116

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase3c"))
sys.path.insert(0, str(HERE.parent / "phase3b"))
from rescore import view, clean, MARKER_RE  # noqa: E402
from scorer import answers_match  # noqa: E402


def extract_answer(text: str):
    """Phase-3c rescore logic, </think> gate removed for Gemma (all arms)."""
    import re
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        return boxed[-1]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    best = None
    for i, l in enumerate(lines):
        lv = view(l)
        m = MARKER_RE.search(lv)
        if m:
            payload = lv[m.end():].lstrip(":").strip()
            if not payload and i + 1 < len(lines):
                payload = view(lines[i + 1]).lstrip(":").strip()
            c = clean(payload)
            if c:
                best = c
    if best:
        return best
    return clean(lines[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--gmu", type=float, default=0.92)
    ap.add_argument("--fp8", action="store_true", help="on-the-fly fp8 (A0 reference arm)")
    ap.add_argument("--moe-backend", default="triton",
                    help="force MoE kernel backend (WSL lacks nvcc for flashinfer JIT)")
    ap.add_argument("--max-model-len", type=int, default=40000)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="drop CUDA graphs to free VRAM (reference arm)")
    ap.add_argument("--cpu-offload-gb", type=float, default=0.0,
                    help="offload N GB of weights to CPU RAM (A0 fp8 reference)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    problems = json.loads((HERE.parent / "phase3c" / "problems.json").read_text())[: args.n]

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": p["q"]}],
                                add_generation_prompt=True, tokenize=False)
        for p in problems
    ]

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gmu,
              moe_backend=args.moe_backend,
              enforce_eager=args.enforce_eager,
              cpu_offload_gb=args.cpu_offload_gb if args.cpu_offload_gb > 0 else None,
              quantization="fp8" if args.fp8 else None)
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64,
                        max_tokens=CAP, seed=SEED)
    outs = llm.generate(prompts, sp)

    rows, texts, correct, answered, cap_hits, tokens_sum = [], {}, 0, 0, 0, 0
    for p, o in zip(problems, outs):
        text = o.outputs[0].text
        n_tok = len(o.outputs[0].token_ids)
        pred = extract_answer(text)
        ok = answers_match(p["gold"], pred)
        rows.append(dict(id=p["id"], gold=p["gold"], pred=pred, ok=ok, n_tokens=n_tok,
                         cap_hit=n_tok >= CAP))
        texts[p["id"]] = text
        correct += ok
        answered += pred is not None
        cap_hits += n_tok >= CAP
        tokens_sum += n_tok

    n = len(rows)
    summary = dict(name=args.name, n=n,
                   accuracy=round(100 * correct / n, 1),
                   answered=round(100 * answered / n, 1),
                   cap_hit_rate=round(100 * cap_hits / n, 1),
                   mean_tokens=round(tokens_sum / n, 1),
                   decoding="card: t=1.0 p=0.95 k=64 seed=20261116",
                   model=args.model, fp8=args.fp8)
    print(json.dumps(summary))
    (HERE / f"{args.name}.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    (HERE / f"{args.name}_texts.json").write_text(json.dumps(texts, indent=1))


if __name__ == "__main__":
    main()
