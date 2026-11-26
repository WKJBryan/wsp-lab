"""Phase 8 runner (inside WSL): phase3c/run_vllm.py plus --no-think, --card-nonthink,
--logit-bias and --force-from. Scores with phase3c/rescore.py; writes phase8/<name>.json
and phase8/<name>_texts.json.gz. Rules: phase8/protocol.md.

  python run_vllm8.py --model ~/wsp3c/stock --name b1 --no-think --card-nonthink
  python run_vllm8.py --model ~/wsp3c/stock --name c1_final --logit-bias 4
  python run_vllm8.py --model ~/wsp3c/stock --name c2_S --force-from p3c_stock

Engine and validity, as in phase3c/run_vllm.py (ISSUES #20): the GPU lock first, prefix caching
off, an optional batch bound (--max-seqs), and the finished run checked by phase3c/runcheck.py.
A run that fails a check is still written, marked as failed inside its file, and then exits
with status 2.
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.set_int_max_str_digits(0)  # top level so the spawned engine process gets it too (ISSUES #15)

HERE = Path(__file__).resolve().parent
P3C = HERE.parent / "phase3c"
sys.path.insert(0, str(P3C))
from gpulock import wait_for_gpu  # noqa: E402
from rescore import judge  # noqa: E402
from runcheck import check_run, refuse_existing, save_run  # noqa: E402
from benchmark import question  # noqa: E402

CAP, SEED, THINK_END = 32768, 20261105, 248069
FORCE_AT, FORCE_NEW = 30720, 2048
CUE = ("\n\nConsidering the limited time by the user, I have to give the solution based on "
       "the thinking directly now.\n</think>\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--card-nonthink", action="store_true", help="model-card non-thinking sampling")
    ap.add_argument("--logit-bias", type=float, default=None, help="constant bias on </think>")
    ap.add_argument("--force-from", default=None, help="phase3c run name to budget-force")
    ap.add_argument("--gmu", type=float, default=0.92)
    ap.add_argument("--max-seqs", type=int, default=0,
                    help="requests in flight at once; 0 = the engine's default (ISSUES #20)")
    ap.add_argument("--reference-question", action="store_true",
                    help="wrap each question as the original authors' evaluation did (calibration)")
    ap.add_argument("--outdir", default=str(HERE),
                    help="where <name>.json and <name>_texts.json.gz are written")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    for existing in (outdir / f"{args.name}.json", outdir / f"{args.name}_texts.json.gz"):
        if existing.exists():
            sys.exit(f"refusing to overwrite {existing}; choose a new --name (ISSUES #20)")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    problems = json.loads((P3C / "problems.json").read_text())[: args.n]
    msgs = [[{"role": "user", "content": question(p, args.reference_question)}] for p in problems]
    think = not args.no_think
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=think) for m in msgs]

    dec = (dict(temperature=1.0, top_p=1.0, presence_penalty=2.0) if args.card_nonthink
           else dict(temperature=0.6, top_p=0.95, presence_penalty=1.5))
    sp = SamplingParams(
        **dec, top_k=20, min_p=0.0, repetition_penalty=1.0, seed=SEED,
        max_tokens=FORCE_NEW if args.force_from else CAP,
        logit_bias=None if args.logit_bias is None else {THINK_END: args.logit_bias},
    )
    wait_for_gpu()
    refuse_existing(outdir / f"{args.name}.json", outdir / f"{args.name}_texts.json.gz")   # again, after the wait
    llm = LLM(model=args.model, max_model_len=40000, gpu_memory_utilization=args.gmu,
              enforce_eager=False, enable_prefix_caching=False,
              max_num_seqs=args.max_seqs or None)
    t0 = time.time()
    extra = {}

    if args.force_from:
        src = args.force_from
        saved = json.loads(gzip.open(P3C / f"{src}_texts.json.gz", "rt", encoding="utf-8").read())
        srows = {r["id"]: r for r in json.loads((P3C / f"{src}.json").read_text())["rows"]}
        cue_ids = tok.encode(CUE, add_special_tokens=False)
        gens, todo, mismatch = [None] * len(problems), [], 0
        for i, p in enumerate(problems):
            ids = tok.encode(saved[p["id"]], add_special_tokens=False)
            mismatch += len(ids) != srows[p["id"]]["n_tokens"]
            head = ids[:FORCE_AT]
            if "</think>" in tok.decode(head):
                gens[i] = (saved[p["id"]], srows[p["id"]]["n_tokens"], srows[p["id"]]["finish"], None)
            else:
                todo.append((i, head))
        outs = llm.generate(
            [{"prompt_token_ids": tok.encode(prompts[i], add_special_tokens=False) + head + cue_ids}
             for i, head in todo], sp)
        for (i, head), o in zip(todo, outs):
            c = o.outputs[0]
            gens[i] = (tok.decode(head) + CUE + c.text,
                       len(head) + len(cue_ids) + len(c.token_ids), c.finish_reason,
                       len(head) + cue_ids.index(THINK_END))
        extra = dict(force_from=src, forced=len(todo),
                     reencode_mismatch_pct=round(100 * mismatch / len(problems), 1))
    else:
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": think})
        gens = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            gens.append((c.text, len(ids), c.finish_reason,
                         ids.index(THINK_END) if THINK_END in ids else None))

    rows, texts = [], {}
    for p, (text, n_tok, finish, think_end_at) in zip(problems, gens):
        ok, pred, rule = judge(p["gold"], text, args.no_think)
        rows.append(dict(id=p["id"], gold=p["gold"], finish=finish, n_tokens=n_tok,
                         closed="</think>" in text, think_end_at=think_end_at,
                         pred_rescored=pred, ok_rescored=ok, rule=rule))
        texts[p["id"]] = text

    failed, validity = check_run(rows, texts, ok_key="ok_rescored")
    n = len(rows)
    summary = dict(
        name=args.name, model=args.model, n=n, settings=vars(args), **extra,
        accuracy=round(100 * sum(r["ok_rescored"] for r in rows) / n, 1),
        answered=round(100 * sum(r["pred_rescored"] is not None for r in rows) / n, 1),
        finished=round(100 * sum(r["finish"] == "stop" for r in rows) / n, 1),
        mean_tokens=round(sum(r["n_tokens"] for r in rows) / n),
        wall_s=round(time.time() - t0, 1), validity=dict(validity, problems=failed),
    )
    print(json.dumps(summary))
    save_run(outdir / f"{args.name}.json", dict(summary=summary, prompt0=prompts[0], rows=rows),
             outdir / f"{args.name}_texts.json.gz", texts, compress=True)
    if failed:
        print("RUN VALIDITY FAILED (ISSUES #20); files kept as evidence and marked: " + "; ".join(failed),
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
