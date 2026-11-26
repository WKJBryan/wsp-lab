"""Phase 12 redo: the tests pre-registered in phase12/protocol_redo.md, on saved data only.

CPU only; numpy and the standard library. No model is loaded and no forward pass is run.
Reads the saved phase-12 atlas files and probe logs, the saved phase-7 activations
(phase7/probe_out, on disk), saved generation runs (phase3c, phase8, phase11) and two weight
files: the Qwen stock weights inside the repository, and the Gemma wound weight file, whose path
is given on the command line and is never recorded.

Every number is printed as it is computed and stored in phase12/redo.json with the input files
(paths relative to the repository) and the SHA-256 of the protocol file.

Any failure stops the run with exit status 2 and a one-line reason on standard error: an input
that is missing or unparsable, a row without a needed field, a flag that is not True or False, a
validity record that is absent or lists a problem, a check that fails. A redo.json left by an
earlier run is marked as superseded when a new run starts, so a stale result can never be read as
the current one.

Self-tests run at the start of every run: the exact helpers on known values, and the input gate
on scratch copies of the inputs, in a temporary folder outside the repository.

Two sections are marked "added in review": they were not pre-registered, are reported and do not
feed the registered verdicts.

Usage:
  python -B phase12/redo.py <repository path> --gemma-weights <the wound's model.safetensors>
"""

import argparse
import hashlib
import io
import itertools
import json
import math
import re
import shutil
import struct
import sys
import tempfile
import traceback
import zipfile
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

SEED = 20261126
BOOT, N_PERM, K_NULL = 1_000_000, 100_000, 4_000
CHUNK = 100_000            # bootstrap resamples drawn at a time (fewer for wide groups: see chunk())
OTHER_SEEDS = 3            # every interval is recomputed with this many other seeds to show how far it moves
BLOCKS = (3, 4, 6)         # layers a block, in the review's check of dependence between layers
FITTING_SIZES = (5, 11)    # texts a loop direction was fitted on: the wound's 5, Qwen's 11
SETS, DRAWS = 5, 400       # independent sets of DRAWS draws of fitting texts: shows how much the numbers move
AUC_DRAWS, TURNS = 150, 60
ANGLES = list(range(0, 91, 5))
NBINS = 25
HALF_WIDTH, BIN = 500, 40            # the saved 25-bin trajectories: -500..+500 tokens, 40 tokens a bin
SMOOTH = 250                         # tokens in the moving average of test Q1-b
SOON = 32                            # "ends at once": generated tokens
EQUIV_MARGIN = 0.5                   # test W2: |difference| below half the layer-to-layer spread
FOCAL_P = 0.05                       # test W1: a layer qualifies when its null p is at or below this
V2_TOP, V2_REF = 12, 0.2             # test V2: loop contrast largest in at least 12 of 24 layers
EMBED = "model.language_model.embed_tokens.weight"
QWEN_WEIGHTS = "phase3/models/stock/model.safetensors-00001-of-00001.safetensors"
SAVED_PCT = dict(qwen={248069: 0.1681}, gemma={1: 0.9937, 106: 0.0756, 50: 0.9783})
ARMS = [                             # name, file, the field that says whether a generation ran into the cap
    ("qwen_stock", "phase3c/p3c_stock_rescored.json", "finish"),
    ("qwen_exit_row_x1_5", "phase3c/p4_mimo_rescored.json", "finish"),
    ("qwen_stock_rerun", "phase3c/ref_stock_rescored.json", "finish"),
    ("qwen_random_noise_same_row", "phase3c/p4_c2_random_rescored.json", "finish"),
    ("qwen_input_side_only", "phase8/d_in.json", "finish"),
    ("gemma_wound", "phase11/ed_pass1.json", "cap_hit"),
    ("gemma_stop_rows_x1_6", "phase11/ed_pass8.json", "cap_hit"),
    ("gemma_final_norm_x0_4", "phase11/ed_pass9.json", "cap_hit"),
    ("gemma_registered_wound", "phase11/editor_a2_attempt1/ed_pass1.json", "cap_hit"),
    ("gemma_registered_pass2", "phase11/editor_a2_attempt1/ed_pass2.json", "cap_hit"),
]
CHECKS = {"soup_judged": "fragment-tail check", "admission_tested": "admission-order test"}
PAIRS = [                           # two arms that must hold the same problems: what they are compared for
    ("the qwen pair of test T2", "qwen_stock", "qwen_exit_row_x1_5"),
    ("the gemma pair of test T2", "gemma_wound", "gemma_stop_rows_x1_6"),
    ("Qwen's second stock run against the exit-row edit", "qwen_stock_rerun", "qwen_exit_row_x1_5"),
    ("the Gemma arm registered in phase 12", "gemma_registered_wound", "gemma_registered_pass2"),
    ("the control qwen_random_noise_same_row", "qwen_stock", "qwen_random_noise_same_row"),
    ("the control qwen_input_side_only", "qwen_stock", "qwen_input_side_only"),
    ("the control gemma_final_norm_x0_4", "gemma_wound", "gemma_final_norm_x0_4"),
    ("the two baseline passes of the wound", "gemma_wound", "gemma_registered_wound"),
]
INPUTS = [
    "phase12/atlas_qwen.json", "phase12/atlas_gemma.json", "phase11/logs/p12_qwen.log",
    "phase11/logs/p12_gemma.log", "phase7/probe_out/timelines.npz", "phase7/probe_out/resid.npz",
    "phase7/sets.json",
] + [file for _, file, _ in ARMS]
SUPERSEDED = dict(status="superseded",
                  note="a new run of redo.py started and did not finish; there is no current result")


# ----------------------------------------------------------------------------- failures
class InputError(Exception):
    """An input is missing, unreadable, incomplete or invalid. There can be no result."""


def fail(reason):
    """Stop without a result: the reason goes to standard error and the exit status is 2."""
    print(f"redo.py: {reason}; stopped, no current redo.json", file=sys.stderr)
    sys.exit(2)


def describe(exc, repo):
    """A failure in one line; repository files are named by their relative paths."""
    if isinstance(exc, InputError):
        return str(exc)
    if isinstance(exc, FileNotFoundError) and exc.filename:
        return f"input file not found: {rel(repo, exc.filename)}"
    frames = traceback.extract_tb(exc.__traceback__)
    return f"{type(exc).__name__}{f' at line {frames[-1].lineno} of this script' if frames else ''}: {exc}"


def rel(repo, path):
    """A path as written in the repository's files: relative, never absolute."""
    try:
        return Path(path).resolve().relative_to(repo).as_posix()
    except ValueError:
        return "<outside the repository; not recorded>"


# ----------------------------------------------------------------------------- small exact helpers
def fnum(x):
    """A float rounded to 6 significant digits: what is printed and what is stored."""
    return float(f"{float(x):.6g}")


def fnums(values):
    return [fnum(v) for v in np.asarray(values).ravel()]


def binom_tail(n, k, p):
    """P(X >= k) for X ~ Binomial(n, p), in closed form."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(math.fsum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def binom_cdf(n, k, p):
    """P(X <= k)."""
    if k < 0:
        return 0.0
    return float(math.fsum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, min(k, n) + 1)))


def sign_test_two_sided(n_pos, n_neg):
    n = n_pos + n_neg
    return 1.0 if n == 0 else min(1.0, 2 * binom_cdf(n, min(n_pos, n_neg), 0.5))


def mcnemar_exact(recovered, lost):
    """Exact McNemar: under no change, recovered and lost are equally likely."""
    n = recovered + lost
    return 1.0 if n == 0 else min(1.0, 2 * binom_cdf(n, min(recovered, lost), 0.5))


def clopper_pearson(k, n, level=0.95):
    """Exact interval for a proportion."""
    a = (1 - level) / 2

    def solve(fn, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            v = fn(mid)
            root_lies_right = (v < target) if increasing else (v > target)
            if root_lies_right:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: binom_tail(n, k, p), a, True)    # P(X >= k) rises with p
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(n, k, p), a, False)    # P(X <= k) falls with p
    return lower, upper


def auc(x, y):
    """P(x > y) + half P(x = y)."""
    return float(((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))


def auc_columns(pos, neg):
    """AUC of each column of pos against the same column of neg, ties counting one half (as auc())."""
    a, b = len(pos), len(neg)
    z = np.concatenate([pos, neg], axis=0).astype(np.float64)
    order = np.argsort(z, axis=0, kind="stable")
    ranked = np.take_along_axis(z, order, axis=0)
    out = np.empty(z.shape[1])
    for c in range(z.shape[1]):
        _, first, runs = np.unique(ranked[:, c], return_index=True, return_counts=True)
        mid = np.repeat(first + (runs + 1) / 2, runs)              # ranks start at 1: equal values get their mean rank
        ranks = np.empty(a + b)
        ranks[order[:, c]] = mid
        out[c] = (ranks[:a].sum() - a * (a + 1) / 2) / (a * b)
    return out


def multiplicities(ix, n):
    """For each index row of a resample: how often each of the n units was drawn."""
    rows = np.arange(len(ix))[:, None]
    return np.bincount((ix + rows * n).ravel(), minlength=len(ix) * n).reshape(len(ix), n)


def resampled_auc(H1, ix, iy):
    """A for each of many resamples, from a comparison matrix H1 (group 1 by group 2) and the index rows
    ix and iy of the values drawn for each group. Equal to auc(x[ix], y[iy]) for every row."""
    n1, n2 = H1.shape
    return (multiplicities(ix, n1) @ H1 * multiplicities(iy, n2)).sum(1) / (n1 * n2)


def boxcar_centred(x, w=SMOOTH):
    """Moving average of w tokens centred on each token, truncated at the ends of the text."""
    t = np.arange(len(x))
    lo, hi = np.maximum(t - w // 2, 0), np.minimum(t + w // 2, len(x))      # w tokens: t - w/2 ... t + w/2 - 1
    cs = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    return (cs[hi] - cs[lo]) / (hi - lo)


def unit(v):
    return v / np.linalg.norm(v)


def unit_rows(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def med_rel_diff(a, b):
    """Median over items of |a - b| / b."""
    return float(np.median(np.abs(a - b) / b))


def chunk(sizes):
    """Bootstrap resamples drawn at a time: CHUNK, or fewer when the groups are wide."""
    return max(1, min(CHUNK, 8_000_000 // sum(sizes)))


def interval(stat, sizes, rng, level=0.95):
    """Percentile interval(s) of a statistic over BOOT bootstrap resamples, and how far each limit moves
    over OTHER_SEEDS other seeds (the Monte Carlo error of the interval).

    stat takes one index array per group, each of shape (draws, group size), and returns one value per
    draw, or one row of values per draw. Returns (limits, moves): a row [low, high] per statistic, and
    the furthest any limit moves in any of the other seeds."""
    def limits(gen):
        parts = []
        for s in range(0, BOOT, chunk(sizes)):
            m = min(chunk(sizes), BOOT - s)
            parts.append(stat(*[gen.integers(0, n, size=(m, n), dtype=np.int32) for n in sizes]))
        v = np.concatenate(parts)
        return np.percentile(v[:, None] if v.ndim == 1 else v, [50 * (1 - level), 50 * (1 + level)], axis=0).T

    own = limits(rng)
    others = [limits(np.random.default_rng(SEED + 7919 * (k + 1))) for k in range(OTHER_SEEDS)]
    move = np.abs(np.array(others) - own[None]).max(axis=(0, 2))
    return (own[0], move[:1]) if own.shape[0] == 1 else (own, move)      # one statistic: [low, high] and one move


def lag1(x):
    """Lag-1 autocorrelation of a series along depth."""
    x = np.asarray(x, float)
    xm = x - x.mean()
    return float((xm[:-1] * xm[1:]).sum() / (xm * xm).sum())


def block_indices(n, length, draws, rng):
    """Indices into 0..n-1 of `draws` resamples of a series in non-overlapping blocks of `length` in depth
    order (the last block may be shorter): as many blocks as the series has are drawn with replacement,
    joined in the order drawn and cut to the length of the series. A shorter block adds only its own
    layers, so a resample that draws it has fewer layers; its row is then padded with -1."""
    table = np.arange(math.ceil(n / length) * length).reshape(-1, length)
    joined = table[rng.integers(0, len(table), size=(draws, len(table)))].reshape(draws, -1)
    order = np.argsort(joined >= n, axis=1, kind="stable")          # entries past the end of the series are dropped
    return np.take_along_axis(np.where(joined < n, joined, -1), order, axis=1)[:, :n]


def gather(x, indices):
    """Values of x at the index rows; -1 (a missing layer) gives NaN."""
    return np.where(indices >= 0, x[np.maximum(indices, 0)], np.nan)


def block_permutation_exact(a, g, length):
    """Every way of handing as many whole blocks (of `length` consecutive layers in depth order) to one
    model as the first series has, all enumerated. Returns how many relabellings give a difference of
    means at least as large as the observed one, and how many relabellings there are."""
    observed = abs(a.mean() - g.mean())
    blocks = [x[i:i + length] for x in (a, g) for i in range(0, len(x), length)]
    reached = total = 0
    for choice in itertools.combinations(range(len(blocks)), math.ceil(len(a) / length)):
        picked = np.concatenate([blocks[i] for i in choice])
        rest = np.concatenate([blocks[i] for i in range(len(blocks)) if i not in choice])
        reached += abs(picked.mean() - rest.mean()) >= observed - 1e-12
        total += 1
    return reached, total


# ----------------------------------------------------------------------------- saved files
def read_bytes(rel_path, repo):
    try:
        return (repo / rel_path).read_bytes()
    except FileNotFoundError:
        raise InputError(f"input file not found: {rel_path}") from None
    except OSError as exc:
        raise InputError(f"input file cannot be read: {rel_path} ({exc.strerror})") from None


def load_json(rel_path, repo):
    try:
        return json.loads(read_bytes(rel_path, repo).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputError(f"input file is not valid JSON: {rel_path} ({exc})") from None


@contextmanager
def load_npz(rel_path, repo):
    try:
        z = np.load(io.BytesIO(read_bytes(rel_path, repo)), allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise InputError(f"input file is not a readable .npz archive: {rel_path} ({type(exc).__name__})") from None
    try:
        yield z
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise InputError(f"input file cannot be read: {rel_path} ({type(exc).__name__})") from None
    finally:
        z.close()


def need(mapping, key, where):
    """A field of a saved file, or a stop if it is not there."""
    try:
        return mapping[key]
    except (KeyError, IndexError, TypeError):
        raise InputError(f"{where}: no {key!r}") from None


def show(title, section, out, key):
    print(f"\n=== {title} ===")
    print(json.dumps(section, indent=1))
    out[key] = section


def show_stored(title, out, path):
    print(f"\n=== {title} ===")
    node = out
    for key in path:
        node = node[key]
    print(json.dumps(node, indent=1))


def all_finite(obj, path=""):
    """Nothing that is not a finite number is ever written to the result file."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            all_finite(value, f"{path}/{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            all_finite(value, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise InputError(f"a value of the result is not a finite number: {path}")


# ----------------------------------------------------------------------------- weight files
class SafeFile:
    """Tensors of one .safetensors file, read on demand with numpy only (BF16 and F32)."""
    DT = {"BF16": ("<u2", 2), "F32": ("<f4", 4)}

    def __init__(self, path):
        self.path = str(path)
        if not Path(path).is_file():
            raise InputError("a weight file given on the command line was not found")
        try:
            with open(self.path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                self.header = json.loads(f.read(n))
        except (OSError, ValueError) as exc:
            raise InputError(f"a weight file cannot be read ({type(exc).__name__})") from None
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def _read(self, offset, count, dt):
        with open(self.path, "rb") as f:
            f.seek(self.base + offset)
            buf = np.fromfile(f, dtype=dt, count=count)
        if len(buf) != count:
            raise InputError(f"short read in a weight file: {len(buf)} of {count} values")
        return buf

    @staticmethod
    def _f32(x):
        return (x.astype(np.uint32) << 16).view(np.float32) if x.dtype == np.uint16 else x

    def tensor(self, name):
        info = need(self.header, name, "a weight file")
        dt, size = self.DT[info["dtype"]]
        a, b = info["data_offsets"]
        return self._f32(self._read(a, (b - a) // size, dt)).reshape(info["shape"])

    def rows(self, name, start, stop):
        """Rows start:stop of a 2-D tensor."""
        (nrow, ncol) = self.header[name]["shape"]
        dt, size = self.DT[self.header[name]["dtype"]]
        a = self.header[name]["data_offsets"][0]
        return self._f32(self._read(a + start * ncol * size, (stop - start) * ncol, dt)).reshape(stop - start, ncol)


def writer_columns(sf, layer, tag):
    """Every column that writes into the state after this layer, each normalised to length 1."""
    p = f"model.language_model.layers.{layer}"
    dense = sf.tensor(p + ".mlp.down_proj.weight")                   # [hidden, intermediate]
    parts = [dense.T]
    if tag == "gemma":
        exp = sf.tensor(p + ".experts.down_proj")                     # [experts, hidden, intermediate]
        parts.append(exp.transpose(0, 2, 1).reshape(-1, exp.shape[1]))   # one row per expert column, as probe_atlas.py
    C = np.concatenate(parts, axis=0).astype(np.float32)
    norm = np.linalg.norm(C, axis=1, keepdims=True)
    if not (np.isfinite(C).all() and (norm > 0).all()):
        raise InputError(f"{tag} layer {layer}: a writer column is non-finite or of length zero")
    return C / norm


def ratio_of(columns, directions):
    """For each direction (a column vector): mean |cosine|, max |cosine|, and max / mean."""
    A = np.abs(columns @ directions)
    mean, mx = A.mean(0), A.max(0)
    return mean, mx, mx / mean


def null_of_layer(columns, rng, k=K_NULL, chunk_=500):
    """Random unit directions against the real columns of one layer."""
    _, hidden = columns.shape
    mean, mx = [], []
    for s in range(0, k, chunk_):
        G = rng.standard_normal((min(chunk_, k - s), hidden)).astype(np.float32)
        G /= np.linalg.norm(G, axis=1, keepdims=True)
        m, x, _ = ratio_of(columns, G.T)
        mean.append(m)
        mx.append(x)
    mean, mx = np.concatenate(mean), np.concatenate(mx)
    ratio = mx / mean
    return dict(mean_alignment=mean, max_alignment=mx, ratio=ratio, ln_ratio_mean=float(np.log(ratio).mean()))


def p_under_null(null_values, observed):
    """(1 + number of null values at or above the observed one) / (number + 1)."""
    return float((1 + (null_values >= observed).sum()) / (len(null_values) + 1))


def norm_percentiles(sf, rows_wanted):
    """Rank percentile of embedding rows among all rows (as probe_atlas.py computed the saved ones)."""
    (n_rows, _) = sf.header[EMBED]["shape"]
    norms = np.empty(n_rows, np.float64)
    for s in range(0, n_rows, 4096):
        x = sf.rows(EMBED, s, min(n_rows, s + 4096)).astype(np.float32)
        norms[s:s + len(x)] = np.sqrt((x * x).sum(1))
    rank = np.argsort(np.argsort(norms))
    return {int(r): float(rank[r] / (n_rows - 1)) for r in rows_wanted}, norms


# ----------------------------------------------------------------------------- Q1
def onset_lines(rel_path, repo):
    """The per-text lines of a probe log: text id, length, onset. A text line that is not complete stops
    the run; it is never skipped."""
    log = read_bytes(rel_path, repo).decode("utf-8", errors="replace")
    found = []
    for line in log.splitlines():
        head = re.match(r"^  (m\d+): (.*)$", line)
        if head:
            body = re.fullmatch(r"T=(\d+) onset=(\d+)", head.group(2))
            if body is None:
                raise InputError(f"{rel_path}: a text line is incomplete and cannot be skipped: {line.strip()!r}")
            found.append((head.group(1), int(body.group(1)), int(body.group(2))))
    if not found:
        raise InputError(f"{rel_path}: no per-text onsets found")
    return found


def q1_curves(repo):
    """4.2: what each saved 25-bin trajectory shows on its own (description, no bar)."""
    res = {}
    edges = -HALF_WIDTH + BIN * np.arange(NBINS + 1)
    for tag in ("qwen", "gemma"):
        where = f"phase12/atlas_{tag}.json"
        traj = np.asarray(need(load_json(where, repo), "trajectory", where), float)
        if len(traj) != NBINS:
            raise InputError(f"{where}: the trajectory has {len(traj)} bins, not {NBINS}")
        if not np.isfinite(traj).all():
            raise InputError(f"{where}: the trajectory holds a value that is not a finite number")
        found = onset_lines(f"phase11/logs/p12_{tag}.log", repo)
        ids = [k for k, _, _ in found]
        if len(set(ids)) != len(ids):
            raise InputError(f"phase11/logs/p12_{tag}.log: a text is logged twice")
        length = np.array([t for _, t, _ in found])
        onset = np.array([o for _, _, o in found])
        if not (length >= onset + HALF_WIDTH).all():
            raise InputError(f"phase11/logs/p12_{tag}.log: a logged text is too short for its trajectory window")
        pre, plateau = traj[8:12].mean(), traj[21:25].mean()
        half = pre + (plateau - pre) / 2
        reached = [i for i in range(8, NBINS) if traj[i] >= half]
        if not reached:                 # out of reach for finite numbers: the largest of bins 8-11 is at least pre
            raise InputError(f"{where}: the trajectory never reaches its half level")
        b = reached[0]
        frac = (float(b) if b == 8 else
                (b - 1) + min(max((half - traj[b - 1]) / (traj[b] - traj[b - 1]), 0.0), 1.0))
        start = -np.minimum(onset, HALF_WIDTH)                 # first relative position each text has
        full = (start[:, None] <= edges[None, :-1]).sum(0)
        part = ((start[:, None] > edges[None, :-1]) & (start[:, None] < edges[None, 1:])).sum(0)
        res[tag] = dict(
            n_texts=len(onset), text_ids=ids,
            onset_tokens_min_median_max=[int(onset.min()), fnum(np.median(onset)), int(onset.max())],
            texts_with_an_onset_under_500_tokens=int((onset < HALF_WIDTH).sum()),
            trajectory=fnums(traj), texts_covering_bin_fully=full.tolist(),
            texts_covering_bin_partly=part.tolist(),
            pre_level_bins_8_to_11=fnum(pre), plateau_bins_21_to_24=fnum(plateau),
            step=fnum(plateau - pre), half_level=fnum(half),
            half_crossing_first_bin_at_or_above=b, half_crossing_fractional_bin=fnum(frac),
            half_crossing_tokens_after_onset=fnum(-HALF_WIDTH + BIN * frac + BIN / 2),
            note="description only: one aggregate curve per model, no interval, cannot fail")
    return res


def timeline(tl, layer, item):
    key = f"proj{layer}_{item['id']}_{item['group']}"
    if key not in tl.files:
        raise InputError(f"phase7/probe_out/timelines.npz has no saved timeline {key}")
    x = tl[key].astype(np.float32)
    if not np.isfinite(x).all():
        raise InputError(f"non-finite values in the saved timeline {key}")
    if len(x) != item["n_ctx"]:
        raise InputError(f"the saved timeline {key} has {len(x)} tokens, sets.json says {item['n_ctx']}")
    return x


def step(x, c):
    """Plateau minus base around an anchor c (4.3)."""
    return float(x[c + 1000:c + 1500].mean(dtype=np.float64) - x[c - 2000:c - 1500].mean(dtype=np.float64))


def lag(x, o):
    """Tokens between the token-level onset o and the first crossing of the half-step level (4.4)."""
    base = x[o - 2000:o - 1500].mean(dtype=np.float64)
    plateau = x[o + 1000:o + 1500].mean(dtype=np.float64)
    if plateau <= base:
        return None, "step not positive"
    y = boxcar_centred(x)
    level = base + (plateau - base) / 2
    lo, hi = o - 1500, o + 1500
    if y[lo] >= level:
        return None, "already at the level when the search window opens"
    hit = np.nonzero(y[lo:hi] >= level)[0]
    if len(hit) == 0:
        return None, "no crossing inside the search window"
    return int(lo + hit[0]) - o, "defined"


def auc_with_null(x, y, rng):
    """A; its one-sided label-shuffling p, with the number of shuffles that reached A; and a 95% bootstrap
    interval over the units of each group, with its movement over other seeds."""
    z = np.concatenate([x, y])
    n1, n2 = len(x), len(y)
    H = ((z[:, None] > z[None, :]) + 0.5 * (z[:, None] == z[None, :])).astype(np.float64)
    obs = auc(x, y)
    reached = 0
    for _ in range(N_PERM):
        perm = rng.permutation(n1 + n2)
        if H[np.ix_(perm[:n1], perm[n1:])].mean() >= obs:
            reached += 1
    (lo, hi), (move,) = interval(lambda ix, iy: resampled_auc(H[:n1, n1:], ix, iy), [n1, n2], rng)
    return obs, (1 + reached) / (N_PERM + 1), reached, lo, hi, move


def q1_qwen(repo):
    """4.3 (Q1-a) and 4.4 (Q1-b): Qwen only, texts as units."""
    with load_npz("phase7/probe_out/timelines.npz", repo) as tl:
        items = need(load_json("phase7/sets.json", repo), "items", "phase7/sets.json")
        stuck = [it for it in items if it["group"] == "stuck"]
        if not stuck:
            raise InputError("phase7/sets.json: no looping texts")
        anchors = sorted({it["onset"] for it in stuck})
        res = dict(anchor_onsets=anchors, n_looping=len(stuck))
        for layer, role in ((19, "primary"), (15, "replication")):
            rng = np.random.default_rng(SEED + layer)
            x_stuck, s_stuck = [], []
            for it in stuck:                                  # the windows of the two tests must fit the text
                x, o = timeline(tl, layer, it), it["onset"]
                if not (type(o) is int and o >= 2000 and len(x) >= o + 1500):
                    raise InputError(f"phase7/sets.json: the looping text {it['id']} has onset {o} and {len(x)} saved "
                                     f"tokens; the windows of tests Q1-a and Q1-b need 2,000 tokens before the onset "
                                     f"and 1,500 after it")
                x_stuck.append(x)
                s_stuck.append(step(x, o))
            s_stuck = np.array(s_stuck)
            sec = dict(role=role, layer=layer, step_looping_median=fnum(np.median(s_stuck)),
                       step_looping_mean=fnum(s_stuck.mean()))
            for name, group in (("long_non_looping", "deep"), ("terminating", "clean")):
                vals, without = [], 0
                for it in (x for x in items if x["group"] == group):
                    x = timeline(tl, layer, it)
                    fits = [c for c in anchors if c + 1500 <= len(x)]
                    if fits:
                        vals.append(float(np.mean([step(x, c) for c in fits])))
                    else:
                        without += 1
                if len(vals) < 2:
                    raise InputError(f"phase7/probe_out/timelines.npz: only {len(vals)} {group} texts are long enough "
                                     f"to host an anchor for layer {layer}")
                a, p, reached, lo, hi, move = auc_with_null(s_stuck, np.array(vals), rng)
                sec[f"vs_{name}"] = dict(
                    n_texts=len(vals), n_texts_without_an_anchor=without,
                    step_median=fnum(np.median(vals)), step_mean=fnum(np.mean(vals)),
                    A=fnum(a), A_ci95=[fnum(lo), fnum(hi)],
                    A_ci95_move_over_other_seeds=fnum(move),
                    perm_shuffles=N_PERM, perm_shuffles_reached=reached, perm_p_one_sided=fnum(p))
            d = sec["vs_long_non_looping"]
            sec["Q1a_verdict"] = ("PASS: the step is specific to looping texts"
                                  if d["A"] >= 0.80 and d["perm_p_one_sided"] < 0.001 else
                                  "FAIL: not shown specific to looping texts")
            found = [(it["id"], *lag(x, it["onset"])) for x, it in zip(x_stuck, stuck)]
            lag_sec = dict(
                n_with_a_lag=int(sum(w == "defined" for _, _, w in found)),
                texts_without_a_lag={i: w for i, _, w in found if w != "defined"})
            lags = np.array([v for _, v, w in found if w == "defined"], float)
            if len(lags):
                (lo, hi), (move,) = interval(lambda ix: np.median(lags[ix], axis=1), [len(lags)], rng)
                n_pos, n_neg = int((lags > 0).sum()), int((lags < 0).sum())
                lag_sec.update(
                    lags_tokens=lags.astype(int).tolist(),
                    lag_text_ids=[i for i, _, w in found if w == "defined"],
                    texts_crossing_more_than_1000_tokens_before_the_onset=[
                        [i, v] for i, v, w in found if w == "defined" and v < -1000],
                    median=fnum(np.median(lags)), ci95=[fnum(lo), fnum(hi)],
                    ci95_move_over_other_seeds=fnum(move), bootstrap_resamples=BOOT,
                    n_positive=n_pos, n_negative=n_neg,
                    sign_test_p_two_sided=fnum(sign_test_two_sided(n_pos, n_neg)),
                    verdict=("LAGS: the state crosses the half-step level after the token-level onset"
                             if lo > 0 else
                             "LEADS: the state crosses it before the token-level onset" if hi < 0 else
                             "NOT DISTINGUISHABLE FROM ZERO"))
            else:
                lag_sec.update(verdict="NO TEXT HAS A DEFINED LAG")
            sec["Q1b_lag_tokens"] = lag_sec
            res[f"layer_{layer}"] = sec
    return res


# ----------------------------------------------------------------------------- Q2
def field(row, key, where):
    try:
        return row[key]
    except KeyError:
        raise InputError(f"{where}: problem {row.get('id')!r} has no {key} field") from None


def boolean(row, key, where):
    """A field that must be True or False. Handed to np.array(..., bool) as it is, any other value
    ("no", 1, None) would become True or False without a word."""
    value = field(row, key, where)
    if type(value) is not bool:
        raise InputError(f"{where}: the {key} of problem {row.get('id')!r} is {value!r}, not True or False")
    return value


def count(row, key, where):
    value = field(row, key, where)
    if type(value) is not int:
        raise InputError(f"{where}: the {key} of problem {row.get('id')!r} is {value!r}, not a whole number")
    return value


def capped_of(row, cap_field, where):
    """Whether a generation ran into the token cap."""
    if cap_field == "cap_hit":
        return boolean(row, "cap_hit", where)
    finish = field(row, "finish", where)
    if finish not in ("stop", "length"):
        raise InputError(f"{where}: the finish of problem {row.get('id')!r} is {finish!r}, neither 'stop' nor 'length'")
    return finish == "length"


def validity_record(record, where):
    """What a saved run says about its own checks. It must say something, and its problems list must be
    empty. Which checks were judged is kept: a run can pass only because some checks never ran."""
    if not isinstance(record, dict):
        raise InputError(f"{where}: no validity record, so the file does not say whether the run passed its checks")
    problems = record.get("problems")
    if not isinstance(problems, list):
        raise InputError(f"{where}: the validity record has no problems list")
    if problems:
        raise InputError(f"{where} failed its own validity check and is not a measurement: {problems}")
    judged, not_judged, unstated = [], [], []
    for flag, name in CHECKS.items():
        if flag not in record:
            unstated.append(name)
        elif type(record[flag]) is not bool:
            raise InputError(f"{where}: {flag} in the validity record is {record[flag]!r}, not True or False")
        else:
            (judged if record[flag] else not_judged).append(name)
    return dict(problems=[], fragment_tail_percent=record.get("soup_pct"), checks_judged=judged,
                checks_not_judged=not_judged, checks_not_stated=unstated)


class Arm:
    """One saved generation run that passed the gate: per problem, whether the generation hit the cap
    and whether the answer was scored right."""

    def __init__(self, rel_path, repo, cap_field):
        data = load_json(rel_path, repo)
        summary, rows = data.get("summary"), data.get("rows")
        if not isinstance(summary, dict) or not isinstance(rows, list) or not rows:
            raise InputError(f"{rel_path}: no summary or no rows")
        self.file = rel_path
        stamp = summary.get("rescored_with")
        if not (isinstance(stamp, str) and stamp.strip()):
            raise InputError(f"{rel_path}: the summary has no rescored_with stamp, so its accuracy cannot be "
                             f"attributed to a checker")
        self.rescored_with, self.summary_accuracy = stamp, summary.get("accuracy")
        self.validity = validity_record(summary.get("validity"), rel_path)
        if not all(isinstance(r, dict) and type(r.get("id")) is str for r in rows):
            raise InputError(f"{rel_path}: a row is not a problem record with an id")
        self.ids = sorted(r["id"] for r in rows)
        if len(set(self.ids)) != len(rows):
            raise InputError(f"{rel_path}: a problem id occurs twice")
        by_id = {r["id"]: r for r in rows}
        scored = "ok_rescored" if "ok_rescored" in rows[0] else "ok"     # one accuracy field for the whole file
        self.capped = np.array([capped_of(by_id[i], cap_field, rel_path) for i in self.ids], bool)
        self.ok = np.array([boolean(by_id[i], scored, rel_path) for i in self.ids], bool)
        self.n_tokens = np.array([count(by_id[i], "n_tokens", rel_path) for i in self.ids], int)
        self.n = len(rows)


def preflight(repo):
    """The gate, run before anything is computed: read every generation-run input, and check that each pair
    of arms holds the same problems. The first input that is unreadable, incomplete or invalid stops the run."""
    arms = {name: Arm(path, repo, cap) for name, path, cap in ARMS}
    for what, a, b in PAIRS:
        if arms[a].ids != arms[b].ids:
            raise InputError(f"{what}: {arms[a].file} and {arms[b].file} do not hold the same problems")
    return arms


def input_checks(arms):
    """For the record: what each generation-run input says about itself."""
    return dict(
        arms={name: dict(file=a.file, rows=a.n, rescored_with=a.rescored_with,
                         accuracy_in_the_summary=a.summary_accuracy, answers_scored_right=int(a.ok.sum()),
                         **a.validity) for name, a in arms.items()},
        not_every_check_judged={
            name: dict(file=a.file, checks_judged=a.validity["checks_judged"],
                       checks_not_judged=a.validity["checks_not_judged"],
                       checks_not_stated=a.validity["checks_not_stated"])
            for name, a in arms.items()
            if a.validity["checks_not_judged"] or a.validity["checks_not_stated"]})


def paired(base, boost, rng):
    """Exit-row boost against the unedited model, paired by problem (the gate has checked that they hold
    the same problems)."""
    c1, c0 = base.capped, boost.capped
    recovered, lost = int((c1 & ~c0).sum()), int((~c1 & c0).sum())
    n = base.n
    capped_before, capped_after = c1.astype(np.int64), c0.astype(np.int64)
    change = boost.ok.astype(np.int64) - base.ok.astype(np.int64)

    def stat(ix):                                            # both intervals from the same resamples
        m = multiplicities(ix, n)
        before = m @ capped_before
        return np.stack([(before - m @ capped_after) / before, 100 * (m @ change) / n], axis=1)

    limits, (r_move, a_move) = interval(stat, [n], rng)          # one row of two statistics per resample
    (r_lo, r_hi), (a_lo, a_hi) = limits
    ended = boost.n_tokens[~c0]
    soon = int((ended <= SOON).sum())
    return dict(
        unedited=base.file, exit_rows_only=boost.file, n_problems=base.n,
        capped_before=int(c1.sum()), capped_after=int(c0.sum()), recovered=recovered, lost=lost,
        lost_problem_ids=[i for i, b, a in zip(base.ids, c1, c0) if not b and a],
        relative_reduction=fnum((c1.sum() - c0.sum()) / c1.sum()),
        relative_reduction_ci95=[fnum(r_lo), fnum(r_hi)], relative_reduction_ci95_move_over_other_seeds=fnum(r_move),
        mcnemar_p_two_sided=fnum(mcnemar_exact(recovered, lost)),
        accuracy_points_before=fnum(100 * base.ok.mean()), accuracy_points_after=fnum(100 * boost.ok.mean()),
        accuracy_change_points=fnum(100 * (boost.ok.mean() - base.ok.mean())),
        accuracy_change_ci95=[fnum(a_lo), fnum(a_hi)], accuracy_change_ci95_move_over_other_seeds=fnum(a_move),
        rescored_with=dict(unedited=base.rescored_with, exit_rows_only=boost.rescored_with),
        bootstrap_resamples=BOOT,
        boosted_arm_ended_within_32_tokens=dict(
            k=soon, n=int(len(ended)), share=fnum(soon / max(1, len(ended))),
            exact_ci95=fnums(clopper_pearson(soon, len(ended))) if len(ended) else None))


def q2(arms):
    """5.1: test T2 and the numbers reported beside it."""
    rng = np.random.default_rng(SEED + 2)
    res = {}
    for model, b, e in (("qwen", "qwen_stock", "qwen_exit_row_x1_5"),
                        ("gemma", "gemma_wound", "gemma_stop_rows_x1_6")):
        res[model] = paired(arms[b], arms[e], rng)
        res[model]["passes_T2_bar"] = bool(res[model]["mcnemar_p_two_sided"] < 0.05
                                           and res[model]["relative_reduction"] >= 0.25)
    res["T2_verdict"] = ("PASS: boosting only the exit rows cuts cap-hits in both models (a necessary "
                         "condition; not evidence that the two failures are the same function)"
                         if res["qwen"]["passes_T2_bar"] and res["gemma"]["passes_T2_bar"] else
                         "FAIL: not shown in both models")
    w, r = arms["gemma_wound"], arms["gemma_registered_wound"]
    res["reported_not_in_the_bar"] = dict(
        qwen_against_the_second_stock_run=paired(arms["qwen_stock_rerun"], arms["qwen_exit_row_x1_5"], rng),
        gemma_arm_registered_in_phase_12=paired(arms["gemma_registered_wound"], arms["gemma_registered_pass2"], rng),
        # the baseline of test T2 is attempt 5's pass 1; phase 12 used attempt 1's pass 1
        gemma_two_baseline_passes=dict(same_capped_outcomes=int((w.capped == r.capped).sum()), of=w.n))
    ctrl = {}                                   # unmatched controls: counts only
    for name, base, other in (("qwen_random_noise_same_row", "qwen_stock", "qwen_random_noise_same_row"),
                              ("qwen_input_side_only", "qwen_stock", "qwen_input_side_only"),
                              ("gemma_final_norm_x0_4", "gemma_wound", "gemma_final_norm_x0_4")):
        c1, c0 = arms[base].capped, arms[other].capped
        ctrl[name] = dict(
            capped_baseline=int(c1.sum()), capped_control=int(c0.sum()),
            fewer_capped=int((c1 & ~c0).sum()), more_capped=int((~c1 & c0).sum()),
            relative_reduction=fnum((c1.sum() - c0.sum()) / c1.sum()),
            note="an unmatched control (another kind of edit or another site); counts only")
    res["unmatched_controls"] = ctrl
    return res


# ----------------------------------------------------------------------------- Q3
def state_means(repo):
    """Mean state at every layer output for each saved class, plus the early/late split of the pre-onset positions."""
    out = {}
    where = "phase7/probe_out/resid.npz"
    with load_npz(where, repo) as R:
        for k in ("loop", "pre", "clean", "deep"):
            out[k] = need(R, f"resid_{k}", where).astype(np.float32).mean(0)                 # [25, 1024]
        meta = need(R, "meta_pre", where)                                                    # item, half, position
        order = np.lexsort((meta[:, 2], meta[:, 0]))
        per_item = Counter(meta[order, 0].tolist())
        if set(per_item.values()) != {32}:
            raise InputError(f"{where}: pre-onset positions per text are not 32 everywhere: "
                             f"{sorted(set(per_item.values()))}")
        X = R["resid_pre"].astype(np.float32)[order]
        early = np.arange(len(X)) % 32 < 16                                      # the 16 lower positions of each text
        out["pre_early"], out["pre_late"] = X[early].mean(0), X[~early].mean(0)
    return out


def reader_check(files):
    """6.4 (a): the saved norm percentiles must come out of the weight files."""
    check = {}
    for tag, sf in files.items():
        got, norms = norm_percentiles(sf, sorted(SAVED_PCT[tag]))
        check[tag] = dict(
            recomputed={str(r): fnum(v) for r, v in got.items()},
            saved={str(r): v for r, v in SAVED_PCT[tag].items()},
            embedding_rows=int(len(norms)),
            row_norm_min_median_max=[fnum(norms.min()), fnum(np.median(norms)), fnum(norms.max())],
            largest_difference=fnum(max(abs(got[r] - SAVED_PCT[tag][r]) for r in got)))
    if not all(c["largest_difference"] <= 0.002 for c in check.values()):
        raise InputError(f"reader check failed: {check}; the saved norm percentiles were not reproduced "
                         f"from the weight files")
    return check


def layer_rows(tag, sf, atlas, rng, contrasts):
    """6.1, 6.2 and 6.4 (b), (c) for one model: one row per layer."""
    where = f"phase12/atlas_{tag}.json"
    hidden = int(sf.header[EMBED]["shape"][1])
    text_book = math.sqrt(2 / (math.pi * hidden))
    rows, n_columns = [], None
    for L in range(atlas["n_layers"]):
        C = writer_columns(sf, L, tag)
        if n_columns is None:
            n_columns = int(C.shape[0])
        elif C.shape[0] != n_columns:
            raise InputError(f"{tag}: layer {L} has {C.shape[0]} writer columns, the others {n_columns}")
        null = null_of_layer(C, rng)
        w = need(atlas["writer_ratios"], str(L), where)
        r_obs = w["max"] / w["mean"]
        # the saved mean, max and ratio are rounded to 5, 5 and 2 decimals: max / mean may differ from
        # the saved ratio by that rounding and by nothing else
        rounding = 0.005 + r_obs * (0.5e-5 / w["mean"] + 0.5e-5 / w["max"])
        if abs(r_obs - w["ratio"]) > rounding + 1e-9:
            raise InputError(f"{where}: layer {L}: saved ratio {w['ratio']} disagrees with saved max / mean "
                             f"{r_obs:.4f}")
        null_mean = float(null["mean_alignment"].mean())
        row = dict(
            layer=L, mean_observed=w["mean"], max_observed=w["max"], ratio_observed=fnum(r_obs),
            null_mean_alignment=fnum(null_mean),
            null_mean_alignment_over_textbook=fnum(null_mean / text_book),
            null_ratio_mean=fnum(null["ratio"].mean()),
            null_ratio_p01=fnum(np.percentile(null["ratio"], 1)),
            null_ratio_p95=fnum(np.percentile(null["ratio"], 95)),
            ln_ratio_null_mean=fnum(null["ln_ratio_mean"]),
            p_mean=fnum(p_under_null(null["mean_alignment"], w["mean"])),
            p_max=fnum(p_under_null(null["max_alignment"], w["max"])),
            p_ratio=fnum(p_under_null(null["ratio"], r_obs)),
            E_ln_excess=fnum(math.log(r_obs) - null["ln_ratio_mean"]))
        if contrasts:
            k = L + 1                                                   # the state this layer's writers produce
            directions = np.stack([unit(a[k] - b[k]) for _, a, b in contrasts], axis=1)
            mean_c, max_c, ratio_c = ratio_of(C, directions)
            row["qwen_rederived"] = dict(mean_alignment=fnum(mean_c[0]), max_alignment=fnum(max_c[0]),
                                         ratio=fnum(ratio_c[0]))
            row["qwen_contrast_ratios"] = {name: fnum(v) for (name, _, _), v in zip(contrasts, ratio_c)}
            row["qwen_loop_contrast_is_largest"] = bool(int(np.argmax(ratio_c)) == 0)
        rows.append(row)
    return rows, n_columns, hidden


def w1(rows, n_layers):
    """6.2: within one model, is the loop direction written more focally than a random direction?"""
    need_ = -(-9 * n_layers // 10)                       # 90% of the layers, rounded up, in whole numbers
    n_focal = int(sum(r["p_ratio"] <= FOCAL_P for r in rows))
    lowest = min(r["null_ratio_p01"] for r in rows)
    p_focal = binom_tail(n_layers, n_focal, FOCAL_P)
    return dict(
        layers_ratio_above_null=n_focal, layers_needed=need_,
        layers_mean_above_null=int(sum(r["p_mean"] <= FOCAL_P for r in rows)),
        layers_max_above_null=int(sum(r["p_max"] <= FOCAL_P for r in rows)),
        binomial_reference_p=fnum(p_focal),
        verdict="FOCAL" if n_focal >= need_ and p_focal < 0.001 else "NOT SHOWN FOCAL",
        registered_diffuse_bar=3.0, lowest_null_first_percentile_over_layers=fnum(lowest),
        registered_bar_below_the_null=bool(3.0 < lowest))


def qwen_checks(rows, contrasts):
    """6.4 (b) the alignment check and (c) test V2, Qwen only."""
    saved_r = np.array([r["ratio_observed"] for r in rows])
    re_r = np.array([r["qwen_rederived"]["ratio"] for r in rows])
    saved_m = np.array([r["mean_observed"] for r in rows])
    re_m = np.array([r["qwen_rederived"]["mean_alignment"] for r in rows])
    r_pearson, d_ratio, d_mean = pearson(saved_r, re_r), med_rel_diff(re_r, saved_r), med_rel_diff(re_m, saved_m)
    n_top = int(sum(r["qwen_loop_contrast_is_largest"] for r in rows))
    p_top = binom_tail(len(rows), n_top, V2_REF)
    v1 = dict(pearson_saved_vs_rederived_ratio=fnum(r_pearson), median_rel_diff_ratio=fnum(d_ratio),
              median_rel_diff_mean=fnum(d_mean),
              verdict="PASS" if r_pearson >= 0.8 and d_ratio <= 0.25 and d_mean <= 0.40 else "FAIL")
    v2 = dict(
        layers_loop_contrast_largest=n_top, layers_needed=V2_TOP, reference_probability=V2_REF,
        binomial_reference_p=fnum(p_top),
        median_ratio_per_contrast={name: fnum(np.median([r["qwen_contrast_ratios"][name] for r in rows]))
                                   for name, _, _ in contrasts},
        verdict="SPECIAL" if n_top >= V2_TOP and p_top < 0.001
        else "NOT DISTINGUISHABLE FROM OTHER STATE CONTRASTS")
    return v1, v2


def w2(E, Lraw, rng, both_focal):
    """6.3: the two models on their excess over their own nulls, layers as units."""
    a, g = E["qwen"], E["gemma"]
    sizes = [len(a), len(g)]
    delta = float(a.mean() - g.mean())
    pooled = float(np.sqrt((a.var(ddof=1) + g.var(ddof=1)) / 2))
    z = np.concatenate([a, g])
    n_q = len(a)
    perm = rng.random((N_PERM, len(z))).argsort(axis=1)             # layer labels shuffled, one layer at a time
    d_perm = z[perm[:, :n_q]].mean(1) - z[perm[:, n_q:]].mean(1)
    reached = int((np.abs(d_perm) >= abs(delta)).sum())

    def standardised(x, y):
        return (x.mean(1) - y.mean(1)) / np.sqrt((x.var(1, ddof=1) + y.var(1, ddof=1)) / 2)

    (lo, hi), (move,) = interval(lambda ia, ib: a[ia].mean(1) - g[ib].mean(1), sizes, rng)
    (d_lo, d_hi), (d_move,) = interval(lambda ia, ib: standardised(a[ia], g[ib]), sizes, rng, level=0.90)
    (lo_raw, hi_raw), (raw_move,) = interval(lambda ia, ib: Lraw["qwen"][ia].mean(1) - Lraw["gemma"][ib].mean(1),
                                             sizes, rng)
    short = [n_q - 1, len(g) - 1]
    (lo1, hi1), (move1,) = interval(lambda ia, ib: a[:-1][ia].mean(1) - g[:-1][ib].mean(1), short, rng)
    res = dict(
        E_mean={t: fnum(E[t].mean()) for t in E},
        E_median={t: fnum(np.median(E[t])) for t in E},
        E_sd=dict(qwen=fnum(a.std(ddof=1)), gemma=fnum(g.std(ddof=1)), pooled=fnum(pooled)),
        delta_ln=fnum(delta), delta_ln_ci95=[fnum(lo), fnum(hi)], delta_ln_ci95_move_over_other_seeds=fnum(move),
        delta_as_a_factor=fnum(math.exp(delta)), factor_ci95=[fnum(math.exp(lo)), fnum(math.exp(hi))],
        perm_shuffles=N_PERM, perm_shuffles_reached=reached,
        perm_p_two_sided=fnum((1 + reached) / (N_PERM + 1)),
        D_standardised=fnum(delta / pooled), D_ci90=[fnum(d_lo), fnum(d_hi)],
        D_ci90_move_over_other_seeds=fnum(d_move), equivalence_margin=EQUIV_MARGIN,
        equivalent=bool(d_lo > -EQUIV_MARGIN and d_hi < EQUIV_MARGIN),
        without_the_null_subtraction=dict(
            delta_ln=fnum(float(Lraw["qwen"].mean() - Lraw["gemma"].mean())),
            as_a_factor=fnum(math.exp(Lraw["qwen"].mean() - Lraw["gemma"].mean())),
            ci95=[fnum(lo_raw), fnum(hi_raw)], ci95_move_over_other_seeds=fnum(raw_move)),
        bootstrap_resamples=BOOT)
    res["verdict"] = ("CORRESPOND" if both_focal and res["equivalent"] else
                      "DIFFER" if lo > 0 or hi < 0 else "INCONCLUSIVE")
    res["sensitivity_without_last_layers"] = dict(
        **{t: dict(E_mean_without_last_layer=fnum(E[t][:-1].mean()), last_layer_E=fnum(E[t][-1])) for t in E},
        delta_ln=fnum(float(a[:-1].mean() - g[:-1].mean())), ci95=[fnum(lo1), fnum(hi1)],
        ci95_move_over_other_seeds=fnum(move1), note="reported, not used for the verdict")
    return res


def layer_dependence(E, rng):
    """Added in review, not pre-registered: does the difference W2 finds survive when neighbouring layers
    are treated as dependent? Layers are resampled, and relabelled, in whole blocks of consecutive depth."""
    a, g = E["qwen"], E["gemma"]
    out = dict(
        note="added in review; reported, not used for the registered verdict",
        lag1_autocorrelation_of_the_excess={
            t: dict(all_layers=fnum(lag1(E[t])), without_the_last_layer=fnum(lag1(E[t][:-1]))) for t in E},
        layer_pairs=dict(qwen_larger=int((a[:, None] > g[None, :]).sum()), of=int(len(a) * len(g))),
        blocks=[])
    for length in BLOCKS:
        d, dst = [], []
        for s in range(0, BOOT, CHUNK):
            x = gather(a, block_indices(len(a), length, min(CHUNK, BOOT - s), rng))
            y = gather(g, block_indices(len(g), length, min(CHUNK, BOOT - s), rng))
            d.append(np.nanmean(x, 1) - np.nanmean(y, 1))
            dst.append(d[-1] / np.sqrt((np.nanvar(x, 1, ddof=1) + np.nanvar(y, 1, ddof=1)) / 2))
        d, dst = np.concatenate(d), np.concatenate(dst)
        reached, total = block_permutation_exact(a, g, length)
        out["blocks"].append(dict(
            layers_a_block=length, qwen_blocks=math.ceil(len(a) / length), wound_blocks=math.ceil(len(g) / length),
            resamples=BOOT,
            difference_ci95=fnums(np.percentile(d, [2.5, 97.5])),
            standardised_difference_ci90=fnums(np.percentile(dst, [5, 95])),
            relabellings_enumerated=int(total), relabellings_reached=int(reached),
            block_permutation_p_two_sided=fnum(reached / total)))
    return out


def per_text_states(R, key, where):
    """(texts x 32 positions, 24 layer outputs, hidden): the states of the 80 looping texts in one class."""
    if f"resid_{key}" not in R.files:
        raise InputError(f"{where}: no resid_{key}")
    meta = R[f"meta_{key}"]
    order = np.lexsort((meta[:, 2], meta[:, 0]))
    X = R[f"resid_{key}"].astype(np.float32)[order]
    texts = np.bincount(meta[order, 0] - meta[order, 0].min())
    if not (texts == 32).all():
        raise InputError(f"{where}: {key} positions per text are not 32 everywhere")
    X = X.reshape(len(texts), 32, X.shape[1], X.shape[2])[:, :, 1:]            # after each of the 24 layers
    return X.reshape(-1, X.shape[2], X.shape[3])


def direction_check(repo, sf, null_ln, rederived_ratio, gap_ln, rng):
    """Added in review, not pre-registered. Could the difference W2 finds come from the number of texts the
    loop direction was fitted on (5 for the wound, 11 for Qwen)? No wound state is saved, so the size of
    the effect is measured on the saved Qwen states, with the layers' real writer columns."""
    where = "phase7/probe_out/resid.npz"
    with load_npz(where, repo) as R:
        XL, XP = (per_text_states(R, key, where) for key in ("loop", "pre"))
    n_states, n_layers, hidden = XL.shape
    n_text = n_states // 32
    four = lambda X: X.reshape(n_text, 32, n_layers, hidden)
    mean_loop, mean_pre = four(XL).mean(1), four(XP).mean(1)                 # each text's mean state, per layer
    d80 = unit_rows(mean_loop.mean(0) - mean_pre.mean(0))                    # one unit direction per layer: 80 texts
    n_draws = SETS * DRAWS
    picks = {k: np.array([rng.choice(n_text, k, replace=False) for _ in range(n_draws)]) for k in FITTING_SIZES}
    ratio_80 = np.empty(n_layers)
    ratio = {k: np.empty((n_layers, n_draws)) for k in FITTING_SIZES}
    angle = {k: np.empty((n_layers, n_draws)) for k in FITTING_SIZES}
    turn = np.empty((n_layers, len(ANGLES)))
    for L in range(n_layers):
        C = writer_columns(sf, L, "qwen")
        ratio_80[L] = ratio_of(C, d80[L:L + 1].T)[2][0]
        for k in FITTING_SIZES:
            d = unit_rows(mean_loop[picks[k], L, :].mean(1) - mean_pre[picks[k], L, :].mean(1))
            ratio[k][L] = ratio_of(C, d.T)[2]
            angle[k][L] = np.degrees(np.arccos(np.clip(d @ d80[L], -1, 1)))
        side = rng.standard_normal((TURNS, hidden)).astype(np.float32)
        side = unit_rows(side - (side @ d80[L])[:, None] * d80[L][None, :])   # at right angles to the 80-text direction
        for i, a in enumerate(ANGLES):                                        # the direction turned by a
            d = math.cos(math.radians(a)) * d80[L] + math.sin(math.radians(a)) * side
            turn[L, i] = np.median(ratio_of(C, unit_rows(d).T)[2]) / ratio_80[L]
    if not np.allclose(ratio_80, rederived_ratio, rtol=1e-4, atol=1e-4):
        raise InputError("the direction check and the layer table disagree on the 80-text ratio of Qwen")
    excess_80 = float((np.log(ratio_80) - null_ln).mean())
    per_draw = {k: (np.log(ratio[k]) - null_ln[:, None]).mean(0).reshape(SETS, DRAWS) for k in FITTING_SIZES}
    error = {k: float(per_draw[k].std(ddof=1) / math.sqrt(n_draws)) for k in FITTING_SIZES}   # over the draws
    step_sets = per_draw[5].mean(1) - per_draw[11].mean(1)                 # the step from 11 to 5 texts, once per set
    mean_step = float(per_draw[5].mean() - per_draw[11].mean())
    falls = per_draw[5] - excess_80                                      # each five-text draw: below the 80-text value
    median_turn = np.median(turn, axis=0)
    share_needed = math.exp(-gap_ln)                  # a direction turned far enough falls to this share of its ratio
    below = [i for i, share in enumerate(median_turn) if share < share_needed]
    turn_step = ANGLES[1] - ANGLES[0]
    crossing = (None if not below or below[0] == 0 else
                ANGLES[below[0] - 1] + turn_step * (median_turn[below[0] - 1] - share_needed)
                / (median_turn[below[0] - 1] - median_turn[below[0]]))

    def fitted(texts):                              # pooled over the texts' tokens, as the probe fitted its directions
        return unit_rows(four(XL)[texts].mean((0, 1)) - four(XP)[texts].mean((0, 1)))

    def scores(X, d):                               # (texts x positions, layers)
        return np.stack([X[:, v, :] @ d[v] for v in range(n_layers)], axis=1)

    def held_out_auc(d, held):                      # every state is scored once; the fitting texts are then left out
        loop = scores(XL, d).reshape(n_text, 32, n_layers)[held]
        pre = scores(XP, d).reshape(n_text, 32, n_layers)[held]
        return float(np.median(auc_columns(loop.reshape(-1, n_layers), pre.reshape(-1, n_layers))))

    in_sample = float(np.median(auc_columns(scores(XL, d80), scores(XP, d80))))
    held_out = {}
    for k in (3,) + FITTING_SIZES:
        draws = []
        for _ in range(AUC_DRAWS):
            order = rng.permutation(n_text)
            draws.append(held_out_auc(fitted(order[:k]), order[k:]))
        held_out[f"{k}_fitting_texts"] = dict(draws=AUC_DRAWS, mean=fnum(np.mean(draws)),
                                              fifth_to_95th_percentile=fnums(np.percentile(draws, [5, 95])))
    relative = {k: (ratio[k] / ratio_80[:, None]).mean(1) for k in FITTING_SIZES}     # per layer: the mean over draws
    return dict(
        note="added in review; Qwen only, since no wound state is saved",
        texts=n_text, positions_per_text=32, sets_of_draws=SETS, draws_a_set=DRAWS,
        ratio_to_the_80_text_value={f"{k}_texts": fnum(math.exp(np.mean(np.log(relative[k]))))
                                    for k in FITTING_SIZES},
        mean_excess={"80_texts": fnum(excess_80), **{f"{k}_texts": fnum(per_draw[k].mean()) for k in FITTING_SIZES},
                     "standard_error_over_draws": {f"{k}_texts": fnum(error[k]) for k in FITTING_SIZES}},
        step_from_11_to_5_texts=fnum(mean_step),
        step_standard_error=fnum(math.sqrt(error[5] ** 2 + error[11] ** 2)),
        step_in_each_set=fnums(step_sets),
        W2_gap=fnum(gap_ln),
        share_of_the_W2_gap=fnum(-mean_step / gap_ln),
        five_text_draws_layer_mean_excess_below_the_80_text_value=dict(
            worst_in_each_set=fnums(falls.min(1)), worst_of_all=fnum(falls.min()),
            fifth_percentile=fnum(np.percentile(falls, 5)), median=fnum(np.median(falls))),
        angle_to_the_80_text_direction_degrees={
            f"{k}_texts": dict(median=fnum(np.median(angle[k])), ninetieth_percentile=fnum(np.percentile(angle[k], 90)))
            for k in FITTING_SIZES},
        ratio_after_a_turn=dict(
            angles_degrees=ANGLES, median_over_layers=fnums(median_turn),
            share_of_its_ratio_the_gap_needs=fnum(share_needed),
            turn_at_which_the_median_falls_that_far_degrees=fnum(crossing) if crossing is not None else None),
        held_out_auc=dict(
            layers="median over the 24 layer outputs",
            direction_from_80_texts_in_sample=fnum(in_sample), **held_out))


def q3(repo, qwen_weights, gemma_weights):
    """6.1-6.4, and the two sections added in review."""
    files = dict(qwen=SafeFile(repo / qwen_weights), gemma=SafeFile(gemma_weights))
    res = dict(null_directions_per_layer=K_NULL, reader_check=reader_check(files),
               file_size_bytes={k: Path(v.path).stat().st_size for k, v in files.items()}, layers={})
    atlases = {tag: load_json(f"phase12/atlas_{tag}.json", repo) for tag in ("qwen", "gemma")}
    means = state_means(repo)
    contrasts = [("loop_minus_pre_onset", means["loop"], means["pre"]),
                 ("early_minus_late_pre_onset", means["pre_early"], means["pre_late"]),
                 ("terminating_minus_pre_onset", means["clean"], means["pre"]),
                 ("long_non_looping_minus_pre_onset", means["deep"], means["pre"]),
                 ("long_non_looping_minus_terminating", means["deep"], means["clean"])]
    for tag in ("qwen", "gemma"):
        rng = np.random.default_rng(SEED + (10 if tag == "qwen" else 11))
        n_layers = need(atlases[tag], "n_layers", f"phase12/atlas_{tag}.json")
        rows, n_columns, hidden = layer_rows(tag, files[tag], atlases[tag], rng,
                                             contrasts if tag == "qwen" else None)
        sec = dict(n_layers=n_layers, n_columns=n_columns, hidden=hidden, layer=rows,
                   W1=w1(rows, n_layers), v1=None, v2=None)
        if tag == "qwen":
            sec["v1"], sec["v2"] = qwen_checks(rows, contrasts)
        res["layers"][tag] = sec
    E = {t: np.array([r["E_ln_excess"] for r in res["layers"][t]["layer"]]) for t in ("qwen", "gemma")}
    Lraw = {t: np.log([r["ratio_observed"] for r in res["layers"][t]["layer"]]) for t in ("qwen", "gemma")}
    both_focal = all(res["layers"][t]["W1"]["verdict"] == "FOCAL" for t in ("qwen", "gemma"))
    res["W2"] = w2(E, Lraw, np.random.default_rng(SEED + 12), both_focal)
    res["layer_dependence"] = layer_dependence(E, np.random.default_rng(SEED + 13))
    qwen = res["layers"]["qwen"]["layer"]
    res["direction_check"] = direction_check(
        repo, files["qwen"], np.array([r["ln_ratio_null_mean"] for r in qwen]),
        np.array([r["qwen_rederived"]["ratio"] for r in qwen]), res["W2"]["delta_ln"], np.random.default_rng(SEED + 14))
    return res


# ----------------------------------------------------------------------------- self-tests
def _self_test_helpers():
    """The helpers that decide verdicts, on known values."""
    assert abs(binom_tail(10, 10, 0.5) - 2 ** -10) < 1e-15
    assert abs(sign_test_two_sided(40, 0) - 2 * 2 ** -40) < 1e-12
    lo, hi = clopper_pearson(1, 64)
    assert abs(lo - (1 - 0.975 ** (1 / 64))) < 1e-10, lo                 # closed form of the lower bound
    assert abs(binom_tail(64, 1, lo) - 0.025) < 1e-9 and abs(binom_cdf(64, 1, hi) - 0.025) < 1e-9, (lo, hi)
    lo, hi = clopper_pearson(0, 64)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 64))) < 1e-12, (lo, hi)
    assert abs(mcnemar_exact(36, 0) - 2 * 2 ** -36) < 1e-10 and mcnemar_exact(0, 0) == 1.0
    assert abs(boxcar_centred(np.arange(1000.0))[500] - 499.5) < 1e-9
    assert auc(np.array([1.0, 3.0]), np.array([2.0])) == 0.5
    r = np.random.default_rng(1)
    x, y = r.normal(size=30), r.normal(size=23) + 3
    assert abs(auc(x, y) + auc(y, x) - 1.0) < 1e-12
    # AUC by midranks (column by column) and AUC from resampling counts, both equal to the pairwise definition
    x, y = r.integers(0, 6, 40).astype(float), r.integers(0, 6, 25).astype(float)       # many ties
    assert abs(auc_columns(x[:, None], y[:, None])[0] - auc(x, y)) < 1e-12
    z = np.concatenate([x, y])
    H = (z[:, None] > z[None, :]) + 0.5 * (z[:, None] == z[None, :])
    ix, iy = r.integers(0, len(x), (7, len(x))), r.integers(0, len(y), (7, len(y)))
    assert np.allclose(resampled_auc(H[:len(x), len(x):], ix, iy), [auc(x[i], y[j]) for i, j in zip(ix, iy)])
    ix, v = r.integers(0, 9, (6, 9)), r.integers(0, 4, 9)           # sums from multiplicities are sums of the draws
    assert (multiplicities(ix, 9) @ v == v[ix].sum(1)).all()
    # the interval helper: shape, and the lag-1 autocorrelation on a known series
    (lo, hi), (move,) = interval(lambda ix: np.median(ix, axis=1), [5], np.random.default_rng(2))
    assert 0 <= lo <= hi <= 4 and move >= 0, (lo, hi, move)
    assert abs(lag1([1, 2, 3, 4]) - 0.25) < 1e-12 and abs(lag1([1, -1, 1, -1]) + 0.75) < 1e-12


def _self_test_blocks():
    """Block resampling: whole blocks in depth order, joined as drawn, a shorter last block adds only its
    own layers, the result is cut to the length of the series, a shorter resample is padded with -1."""
    for n, length in ((24, 3), (30, 4), (10, 4), (7, 2)):
        for row in block_indices(n, length, 200, np.random.default_rng(4)):
            valid = row[row >= 0]
            assert len(valid) > 0 and row[:len(valid)].tolist() == valid.tolist()       # the layers, then the padding
            p = 0
            while p < len(valid):
                first = int(valid[p]) // length * length
                segment = list(range(first, min(first + length, n)))
                assert valid[p:p + len(segment)].tolist() == segment[:len(valid) - p], (n, length, valid)
                p += len(segment)
    values = np.arange(10.0)
    padded = gather(values, np.array([[3, 4, -1, -1], [9, 0, 1, 2]]))
    assert np.isnan(padded[0, 2]) and np.nanmean(padded, 1).tolist() == [3.5, 3.0]
    reached, total = block_permutation_exact(np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0]), 2)
    assert total == math.comb(4, 2) and reached >= 1, (reached, total)     # 4 blocks, 2 handed to the first series


def _self_test_gate(repo):
    """The input gate and the entry point, on scratch copies of the inputs in a temporary folder outside
    the repository. Each fault must stop the run with exit status 2 and leave no current result.

    Returns the faults tried, or None when an input cannot be copied or does not pass the gate as it is:
    then there is nothing to inject a fault into, and the run's own gate reports that input."""
    qwen, gemma = "phase3c/p3c_stock_rescored.json", "phase11/ed_pass8.json"
    with tempfile.TemporaryDirectory(prefix="wsp-redo-self-test-") as tmp:
        scratch = Path(tmp)
        try:
            for rel_path in [file for _, file, _ in ARMS] + ["phase12/protocol_redo.md"]:
                (scratch / rel_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo / rel_path, scratch / rel_path)
            preflight(scratch)                 # the untouched copies must pass the gate
        except (InputError, OSError):
            return None
        target = scratch / "phase12" / "redo.json"

        def restore(rel_path):
            shutil.copy2(repo / rel_path, scratch / rel_path)

        def edit(rel_path, change):
            data = json.loads((scratch / rel_path).read_text(encoding="utf-8"))
            change(data)
            (scratch / rel_path).write_text(json.dumps(data), encoding="utf-8")

        def run_stops(label, words):
            """The whole entry point on the scratch tree: exit status 2, the reason on standard error,
            and the earlier result marked as superseded."""
            target.write_text('{"status": "result of an earlier run"}', encoding="utf-8")
            argv, own = sys.argv, globals()["_self_test"]
            sys.argv = ["redo.py", str(scratch), "--gemma-weights", "not-needed.safetensors"]
            globals()["_self_test"] = lambda repo: None            # the scratch run must not start the self-tests again
            said = io.StringIO()
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(said):
                    try:
                        main()
                    except SystemExit as exc:
                        code = exc.code
                    else:
                        raise AssertionError(f"the run went on despite {label}")
            finally:
                sys.argv, globals()["_self_test"] = argv, own
            assert code == 2, (label, code)
            assert words in said.getvalue(), (label, said.getvalue())
            assert json.loads(target.read_text(encoding="utf-8"))["status"] == "superseded", label
            return label

        cases = [
            ("a validity record that lists a problem", qwen, "failed its own validity check",
             lambda d: d["summary"]["validity"]["problems"].append("injected by the self-test")),
            ("a validity record that is absent", qwen, "no validity record",
             lambda d: d["summary"].pop("validity")),
            ("a validity record that is null", gemma, "no validity record",
             lambda d: d["summary"].__setitem__("validity", None)),
            ("a validity record without a problems list", gemma, "no problems list",
             lambda d: d["summary"]["validity"].pop("problems")),
            ("a cap flag that is not True or False", gemma, "not True or False",
             lambda d: d["rows"][1].__setitem__("cap_hit", "no")),
            ("an accuracy flag that is not True or False", qwen, "not True or False",
             lambda d: d["rows"][1].__setitem__("ok_rescored", "yes")),
            ("a row without a needed field", qwen, "has no finish field",
             lambda d: d["rows"][0].pop("finish")),
            ("a file without a re-score stamp", gemma, "no rescored_with stamp",
             lambda d: d["summary"].pop("rescored_with")),
        ]
        stopped = []
        for label, rel_path, words, change in cases:
            edit(rel_path, change)
            stopped.append(run_stops(label, words))
            restore(rel_path)
        (scratch / gemma).unlink()
        stopped.append(run_stops("an input file that is missing", "input file not found"))
        restore(gemma)
        (scratch / qwen).write_text("{not json", encoding="utf-8")
        stopped.append(run_stops("an input file that is not JSON", "not valid JSON"))
        restore(qwen)
        stopped.append(run_stops("an input needed after the gate is missing", "input file not found"))
        return stopped


def _self_test(repo):
    """Run at the start of every run. A failed self-test stops the run."""
    _self_test_helpers()
    _self_test_blocks()
    stopped = _self_test_gate(repo)
    if stopped is None:
        print("self-tests: helpers and block resampling passed; the input gate was not exercised on scratch "
              "copies because an input is missing or invalid as it is, and the run's own gate reports it")
        return
    print(f"self-tests passed: helpers, block resampling, and the input gate on scratch copies ({len(stopped)} "
          f"faults each stopped the run with exit status 2)")


# ----------------------------------------------------------------------------- main
def compute(repo, qwen_weights, gemma_weights, target):
    """Read every generation-run input (the gate), compute, write. Raises on any failure; main() turns
    that into exit status 2. main() has already marked an older result as superseded."""
    arms = preflight(repo)
    out = dict(
        date="2026-11-26", protocol="phase12/protocol_redo.md",
        protocol_sha256=hashlib.sha256(read_bytes("phase12/protocol_redo.md", repo)).hexdigest(),
        settings=dict(seed=SEED, bootstrap_resamples=BOOT, other_seeds_per_interval=OTHER_SEEDS,
                      permutations=N_PERM, null_directions_per_layer=K_NULL),
        inputs=dict(qwen_weights=(QWEN_WEIGHTS if Path(qwen_weights).name == Path(QWEN_WEIGHTS).name else
                                  "path given on the command line, outside the repository; not recorded"),
                    gemma_weights="path given on the command line, outside the repository; not recorded",
                    saved=INPUTS,
                    sha256_first_16={name: hashlib.sha256(read_bytes(name, repo)).hexdigest()[:16] for name in INPUTS}))
    print(f"protocol {out['protocol']} sha256 {out['protocol_sha256']}")
    print(f"input files hashed: {len(INPUTS)}")
    show("Inputs: validity records and checker stamps of the generation runs", input_checks(arms), out, "input_checks")
    show("4.2  Q1: the two saved trajectories (description only)", q1_curves(repo), out, "Q1_curves")
    show("4.3 / 4.4  Q1, Qwen only: step specificity (Q1-a) and lag (Q1-b)", q1_qwen(repo), out, "Q1_qwen")
    show("5.1  Q2: exit rows only, paired by problem (T2)", q2(arms), out, "Q2")
    out["Q3"] = q3(repo, qwen_weights, gemma_weights)
    show_stored("6.4 (a)  Q3: reader check", out, ("Q3", "reader_check"))
    show_stored("6.1, 6.2, 6.4 (b), (c)  Q3: the null, W1 and the Qwen-only checks", out, ("Q3", "layers"))
    show_stored("6.3  Q3: W2", out, ("Q3", "W2"))
    show_stored("review  Q3: dependence between layers (not pre-registered)", out, ("Q3", "layer_dependence"))
    show_stored("review  Q3: number of fitting texts (not pre-registered; Qwen only)", out, ("Q3", "direction_check"))
    all_finite(out)
    target.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("\n=== verdict lines ===")
    print("Q1 across models: NOT TESTABLE from saved data (one aggregate curve per model)")
    for key in ("layer_19", "layer_15"):
        s = out["Q1_qwen"][key]
        print(f"Q1-a Qwen {key}: {s['Q1a_verdict']} | Q1-b Qwen {key}: {s['Q1b_lag_tokens']['verdict']}")
    print(f"Q2 T2: {out['Q2']['T2_verdict']}")
    for tag in ("qwen", "gemma"):
        print(f"Q3 W1 {tag}: {out['Q3']['layers'][tag]['W1']['verdict']}")
    table = out["Q3"]["W2"]
    print(f"Q3 W2: {table['verdict']} ({table['perm_shuffles_reached']} of {table['perm_shuffles']} layer-label "
          f"shuffles reached the observed difference)")
    print(f"Q3 V2 (Qwen only): {out['Q3']['layers']['qwen']['v2']['verdict']} | "
          f"alignment check: {out['Q3']['layers']['qwen']['v1']['verdict']}")
    print("\n=== accuracy (answers scored right; not part of any bar) ===")
    stamps = sorted({a.rescored_with for a in arms.values()})
    print(f"checker stamp in the files read (rescored_with): {stamps if len(stamps) > 1 else stamps[0]}")
    for model, label in (("qwen", "Qwen: stock against the </think> row x 1.50"),
                         ("gemma", "wound: unedited against the three stop rows x 1.6")):
        s = out["Q2"][model]
        print(f"accuracy {model} ({label}; {s['n_problems']} problems): {s['accuracy_points_before']} -> "
              f"{s['accuracy_points_after']} of 100, change {s['accuracy_change_points']:+} points "
              f"(95% interval {s['accuracy_change_ci95'][0]:+} to {s['accuracy_change_ci95'][1]:+})")
    r = out["Q2"]["reported_not_in_the_bar"]
    print(f"accuracy qwen, second stock run against the same edit: "
          f"{r['qwen_against_the_second_stock_run']['accuracy_points_before']} -> "
          f"{r['qwen_against_the_second_stock_run']['accuracy_points_after']}; accuracy gemma, the arm registered "
          f"in phase 12: {r['gemma_arm_registered_in_phase_12']['accuracy_points_before']} -> "
          f"{r['gemma_arm_registered_in_phase_12']['accuracy_points_after']}")
    print("written: phase12/redo.json")


def main():
    ap = argparse.ArgumentParser(description="Phase-12 redo on saved data (see phase12/protocol_redo.md).")
    ap.add_argument("repo", help="path of the repository")
    ap.add_argument("--gemma-weights", required=True,
                    help="the Gemma wound's model.safetensors (its path is not recorded)")
    ap.add_argument("--qwen-weights", default=QWEN_WEIGHTS,
                    help="the Qwen stock weight file (default: the copy in the repository)")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    protocol, target = repo / "phase12/protocol_redo.md", repo / "phase12/redo.json"
    try:
        if target.exists():      # an older result must not outlive a run that then stops, wherever it stops
            target.write_text(json.dumps(SUPERSEDED, indent=1), encoding="utf-8")
        if not protocol.is_file():
            raise InputError("phase12/protocol_redo.md not found: nothing may be computed before the protocol exists")
        _self_test(repo)
        compute(repo, args.qwen_weights, args.gemma_weights, target)
    except Exception as exc:                # every failure leaves by the same door: status 2, no result
        fail(describe(exc, repo))


if __name__ == "__main__":
    main()
