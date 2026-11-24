"""Phase 11 evaluator: MATH-500 on Gemma-4 arms, vLLM, run INSIDE WSL2.

Decoding = Gemma-4 model card for every arm (generation_config.json):
temperature 1.0, top_p 0.95, top_k 64, 1 sample, cap 32768, fixed seed.
Scoring = phase3c/rescore.py's extractor with no think gate (Gemma has no
think blocks), identical for every arm.

Engine, validity and files (ISSUES #20): the project's GPU lock is taken first,
prefix caching is off, and the finished run is checked by phase3c/runcheck.py
(outcome by position in the batch is a failure; the fragment rate is only
reported, because a badly damaged model may produce such tails itself). An
existing result file is never overwritten: pick a new --name, or --outdir.

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

CAP = 8192  # Amendment #1 (2026-11-16): 32768 -> 8192, identical for every arm.
            # Reason: A0 fp8 weights + one 34K-token KV sequence (3.97 GiB) exceed
            # 32 GB VRAM; vLLM 0.33 cpu_offload is broken for Gemma4 (meta-tensor
            # bug). Healthy solutions run 1-3K tokens; the wound loops at any cap.
SEED = 20261116

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase3c"))
from gpulock import wait_for_gpu  # noqa: E402
from rescore import answers_match, extract_answer  # noqa: E402
from runcheck import check_run, refuse_existing, save_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--outdir", default=str(HERE),
                    help="where <name>.json and <name>_texts.json are written")
    ap.add_argument("--gmu", type=float, default=0.92)
    ap.add_argument("--fp8", action="store_true", help="on-the-fly fp8 (A0 reference arm)")
    ap.add_argument("--moe-backend", default="triton",
                    help="force MoE kernel backend (WSL lacks nvcc for flashinfer JIT)")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="default: CAP + 700")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="drop CUDA graphs to free VRAM (reference arm)")
    ap.add_argument("--cpu-offload-gb", type=float, default=0.0,
                    help="offload N GB of weights to CPU RAM (A0 fp8 reference)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    for existing in (outdir / f"{args.name}.json", outdir / f"{args.name}_texts.json"):
        if existing.exists():
            sys.exit(f"refusing to overwrite {existing}; choose a new --name (ISSUES #20)")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    problems = json.loads((HERE.parent / "phase3c" / "problems.json").read_text())[: args.n]

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": p["q"]}],
                                add_generation_prompt=True, tokenize=False)
        for p in problems
    ]

    max_len = args.max_model_len or (CAP + 700)
    llm_kwargs = dict(model=args.model, max_model_len=max_len,
                      gpu_memory_utilization=args.gmu,
                      moe_backend=args.moe_backend,
                      enforce_eager=args.enforce_eager,
                      enable_prefix_caching=False,
                      max_num_batched_tokens=3072,
                      limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    if args.cpu_offload_gb > 0:
        llm_kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    if args.fp8:
        llm_kwargs["quantization"] = "fp8"
    wait_for_gpu()
    refuse_existing(outdir / f"{args.name}.json", outdir / f"{args.name}_texts.json")   # again, after the wait
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64,
                        max_tokens=CAP, seed=SEED)
    outs = llm.generate(prompts, sp)

    rows, texts, correct, answered, cap_hits, tokens_sum = [], {}, 0, 0, 0, 0
    for p, o in zip(problems, outs):
        text = o.outputs[0].text
        n_tok = len(o.outputs[0].token_ids)
        finish = o.outputs[0].finish_reason
        pred = extract_answer(text, no_think=True)
        ok = answers_match(p["gold"], pred)
        rows.append(dict(id=p["id"], gold=p["gold"], pred=pred, ok=ok, n_tokens=n_tok,
                         finish=finish, cap_hit=n_tok >= CAP))
        texts[p["id"]] = text
        correct += ok
        answered += pred is not None
        cap_hits += n_tok >= CAP
        tokens_sum += n_tok

    failed, validity = check_run(rows, texts, ok_key="ok", soup_bar=None)
    n = len(rows)
    summary = dict(name=args.name, n=n,
                   accuracy=round(100 * correct / n, 1),
                   answered=round(100 * answered / n, 1),
                   cap_hit_rate=round(100 * cap_hits / n, 1),
                   mean_tokens=round(tokens_sum / n, 1),
                   decoding="card: t=1.0 p=0.95 k=64 seed=20261116",
                   model=args.model, fp8=args.fp8, validity=dict(validity, problems=failed))
    print(json.dumps(summary))
    save_run(outdir / f"{args.name}.json", dict(summary=summary, rows=rows),
             outdir / f"{args.name}_texts.json", texts)
    if failed:
        print("RUN VALIDITY FAILED (ISSUES #20); files kept as evidence and marked: " + "; ".join(failed),
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
