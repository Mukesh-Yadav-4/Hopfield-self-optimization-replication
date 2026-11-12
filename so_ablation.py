"""
so_ablation.py  --  2x2(+) ablation on top of so_hopfield.py:  reset type  x  learning rate.

Question: is the energy gain of "partial resets" (a) iterated local search, (b) Hebbian self-optimization, or
(c) an interaction between the two?

Design (full factorial, same W0 instances and same run seeds in every cell)
    reset type : full     = every node re-randomised at each reset                (the paper's protocol)
                 partial  = a random fraction f of units re-randomised, rest keep the previous state
    learning   : alpha = 0 (control)  and  each alpha in --alphas  (default 3e-7, 7e-7)
Protocol per run = so_hopfield's BL / L / AL phases (R resets each, T async steps per reset); the network state
carries across resets (and phases) in the partial arm. Energy is always measured against W0.

Unit of analysis = one W0 instance (a random modular matrix). Results are averaged over --run-seeds inside each W0,
then compared PAIRED across W0 instances (so n = number of W0 seeds, not number of resets).

Primary outcome   : E_AL = mean energy over the AL phase (steady-state quality), lower is better.
Secondary outcomes: E_min = lowest energy seen anywhere in the run (best found), AL modal share, ||W_L - W0||_F.

Contrasts (negative = lower energy = better):
    L_full      = E_AL(full, a)    - E_AL(full, 0)       learning effect under full resets   (sanity: should be < 0)
    L_partial   = E_AL(partial, a) - E_AL(partial, 0)    learning effect under partial resets
    P0          = E_AL(partial, 0) - E_AL(full, 0)       reset effect without learning       (iterated local search)
    Interaction = L_partial - L_full                     >0: partial resets weaken the benefit of learning

Pre-stated expectations (write down BEFORE running, so the result is a test, not a story):
    H1  L_full < 0 for the chosen alphas (else the baseline is not reproducing, stop and debug).
    H2  P0 < 0 (partial resets help even with alpha = 0; the earlier single-seed C-vs-A result).
    H3  Interaction is the open question. One mechanism to test, not assume: SO learns from the DIVERSITY of
        attractors visited after random resets; partial resets keep revisiting one basin, so Hebbian learning
        may entrench it (Interaction > 0).

Caveats built in: alpha is fixed across W0 instances although the alpha->regime map depends on W0 (the per-W0
regime counts for the full-reset arm are printed); "partial reset" here = random units re-randomised (--unit nodes)
or whole modules re-randomised (--unit modules); these are operationalisations, not the paper's definition.

Run:
    python so_ablation.py --selftest-only
    python so_ablation.py                                   # defaults: 10 W0 seeds x 2 run seeds, ~4-5 min
    python so_ablation.py --w-seeds 3 --run-seeds 1 --resets 200      # quick look
    python so_ablation.py --unit modules                    # module-wise partial reset variant
"""
import argparse
import csv
import time

import numpy as np
from numba import njit

from so_hopfield import (_seed, assert_valid, async_step, classify, energy, hebb_update, make_W0,
                         run_protocol)


# --------------------------------------------------------------------------- kernels
@njit(cache=True)
def apply_reset(s, frac, group):
    """Re-randomise a fraction `frac` of the units in place. A unit is a group of `group` consecutive nodes
    (group = 1: single nodes; group = k: whole modules). frac >= 1 re-randomises everything, drawing random
    numbers in node order exactly like so_hopfield.random_state (so the full arm matches that code bit for bit)."""
    N = s.shape[0]
    n_groups = N // group
    m = int(np.floor(frac * n_groups + 0.5))
    if m >= n_groups:
        for i in range(N):
            s[i] = 1.0 if np.random.random() < 0.5 else -1.0
    elif m > 0:
        perm = np.random.permutation(n_groups)
        for q in range(m):
            g0 = perm[q] * group
            for i in range(g0, g0 + group):
                s[i] = 1.0 if np.random.random() < 0.5 else -1.0


@njit(cache=True)
def run_resets_partial(W_L, W0, s, n_resets, T, alpha, frac, group, first_full, E_out, dfro_out, offset):
    """Like so_hopfield.run_resets, but the state s persists across resets and only `frac` of the units are
    re-randomised at each reset (the very first reset of a run, if first_full, randomises everything)."""
    N = W_L.shape[0]
    learn = alpha != 0.0
    for r in range(n_resets):
        if first_full and r == 0:
            apply_reset(s, 1.0, group)
        else:
            apply_reset(s, frac, group)
        for t in range(T):
            async_step(W_L, s)
            if learn:
                hebb_update(W_L, s, alpha)
        E_out[offset + r] = energy(W0, s)
        fro = 0.0
        for a in range(N):
            for b in range(N):
                d = W_L[a, b] - W0[a, b]
                fro += d * d
        dfro_out[offset + r] = np.sqrt(fro)


def run_protocol_partial(W0, alpha, frac, group, n_resets, T, seed):
    """BL (alpha=0) -> L (alpha) -> AL (frozen). frac = 1.0 reproduces so_hopfield.run_protocol exactly."""
    _seed(seed)
    W_L = W0.copy()
    s = np.zeros(W0.shape[0])
    E = np.empty(3 * n_resets)
    dfro = np.empty(3 * n_resets)
    run_resets_partial(W_L, W0, s, n_resets, T, 0.0, frac, group, True, E, dfro, 0)
    run_resets_partial(W_L, W0, s, n_resets, T, alpha, frac, group, False, E, dfro, n_resets)
    run_resets_partial(W_L, W0, s, n_resets, T, 0.0, frac, group, False, E, dfro, 2 * n_resets)
    assert_valid(W_L, "W_L after partial-reset protocol")
    return E, dfro


def summarize(E, dfro, R):
    AL, BL = E[2 * R:], E[:R]
    vals, counts = np.unique(np.round(AL, 6), return_counts=True)
    k = counts.argmax()
    return dict(E_BL=float(BL.mean()), E_AL=float(AL.mean()), E_min=float(E.min()),
                AL_mode=float(vals[k]), AL_modal_frac=float(counts[k] / AL.size), dW=float(dfro[-1]))


# --------------------------------------------------------------------------- self tests
def selftest(k=5):
    print("[selftest] running ...")
    N = 100
    # 1) apply_reset touches at most m units, and whole modules when group = k
    _seed(5)
    for group, frac in [(1, 0.25), (k, 0.25), (1, 0.5), (k, 0.4)]:
        s0 = np.where(np.random.default_rng(1).random(N) < 0.5, -1.0, 1.0)
        s1 = s0.copy()
        apply_reset(s1, frac, group)
        changed = np.flatnonzero(s1 != s0)
        n_groups = N // group
        m = int(np.floor(frac * n_groups + 0.5))
        touched_groups = np.unique(changed // group)
        assert len(touched_groups) <= m, f"apply_reset touched {len(touched_groups)} > {m} units (group={group})"
    # frac = 1 must be able to change (nearly) everything; frac = 0 must change nothing
    s1 = s0.copy(); apply_reset(s1, 0.0, 1); assert np.all(s1 == s0), "frac=0 must be a no-op"

    # 2) full-reset arm == so_hopfield.run_protocol, bit for bit (regression test of the verified baseline)
    W0 = make_W0(N, k, 0.1, 0)
    R, T, a, seed = 20, 300, 1e-5, 3
    E_ref, dfro_ref, _, _ = run_protocol(W0, a, R, T, seed)
    E_new, dfro_new = run_protocol_partial(W0, a, 1.0, 1, R, T, seed)
    assert np.array_equal(E_ref, E_new), "full-reset arm deviates from the verified so_hopfield protocol"
    assert np.allclose(dfro_ref, dfro_new, rtol=0, atol=1e-12), "||dW|| deviates from so_hopfield protocol"

    # 3) partial arm: alpha=0 never changes W_L, alpha>0 does, AL frozen
    E0, d0 = run_protocol_partial(W0, 0.0, 0.25, 1, R, T, seed)
    assert np.all(d0 == 0.0), "alpha = 0 changed W_L in the partial arm"
    E1, d1 = run_protocol_partial(W0, a, 0.25, 1, R, T, seed)
    assert d1[2 * R - 1] > 0.0 and np.ptp(d1[2 * R:]) == 0.0 and np.all(d1[:R] == 0.0), "learning bookkeeping broken"
    print("[selftest] OK  (apply_reset ok; full arm identical to so_hopfield.run_protocol; learning bookkeeping ok)")


# --------------------------------------------------------------------------- statistics
def boot_ci(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])


def wilcoxon_p(d):
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def report_contrast(name, d):
    d = np.asarray(d, dtype=float)
    lo, hi = boot_ci(d)
    print(f"  {name:<44s} mean {d.mean():+8.3f}   95% CI [{lo:+8.3f}, {hi:+8.3f}]   "
          f"units<0: {int((d < 0).sum())}/{len(d)}   Wilcoxon p={wilcoxon_p(d):.3g}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphas", type=float, nargs="+", default=[3e-7, 7e-7])
    ap.add_argument("--frac", type=float, default=0.25, help="fraction of units re-randomised in the partial arm")
    ap.add_argument("--unit", choices=["nodes", "modules"], default="nodes")
    ap.add_argument("--w-seeds", type=int, default=10, help="number of W0 instances (seeds 0..n-1)")
    ap.add_argument("--run-seeds", type=int, default=2, help="run seeds per W0 (averaged within W0)")
    ap.add_argument("--resets", type=int, default=1000, help="resets per phase")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="so_ablation")
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()

    if not args.skip_selftest:
        selftest(args.k)
    if args.selftest_only:
        return

    group = 1 if args.unit == "nodes" else args.k
    arms = [("full", 1.0), ("partial", args.frac)]
    alphas = [0.0] + list(args.alphas)
    R, T = args.resets, args.steps
    rows = []
    regimes = {a: [] for a in args.alphas}      # per-W0 regime letter in the full-reset arm
    print(f"\nW0 seeds 0..{args.w_seeds - 1} x run seeds {args.run_seeds}; partial = {args.frac:.0%} of {args.unit}; "
          f"alphas {alphas}; R={R}, T={T}")
    t0 = time.time()
    for w in range(args.w_seeds):
        W0 = make_W0(args.N, args.k, args.p, w)
        for rs in range(args.run_seeds):
            seed = 1 + rs
            for arm, frac in arms:
                for a in alphas:
                    E, dfro = run_protocol_partial(W0, a, frac, group, R, T, seed)
                    row = dict(w=w, run_seed=seed, reset=arm, alpha=a, **summarize(E, dfro, R))
                    rows.append(row)
                    if arm == "full" and a > 0 and rs == 0:
                        regimes[a].append(classify(E, R)["label"][0])
        print(f"  W0 {w} done  ({time.time() - t0:.0f}s elapsed)")

    with open(f"{args.out}_runs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # aggregate: mean over run seeds within each W0 -> arrays over W0 instances
    def get(arm, a, key):
        out = []
        for w in range(args.w_seeds):
            out.append(np.mean([r[key] for r in rows if r["w"] == w and r["reset"] == arm and r["alpha"] == a]))
        return np.array(out)

    print("\n=== Cell means across W0 instances (mean +- SD over W0) ===")
    print(f"{'reset':<8s}{'alpha':>10s}{'E_AL':>18s}{'E_min':>18s}{'AL modal share':>16s}{'||dW||_F':>12s}")
    for arm, _ in arms:
        for a in alphas:
            e, m, f, d = get(arm, a, "E_AL"), get(arm, a, "E_min"), get(arm, a, "AL_modal_frac"), get(arm, a, "dW")
            print(f"{arm:<8s}{a:>10g}{e.mean():>10.2f} +-{e.std(ddof=1):>5.2f}{m.mean():>10.2f} +-{m.std(ddof=1):>5.2f}"
                  f"{f.mean():>16.2f}{d.mean():>12.2f}")

    print("\n=== Full-reset arm: regime per W0 (A none, B above range, C inside range, D below range, ? other) ===")
    for a in args.alphas:
        letters = regimes[a]
        print(f"  alpha {a:g}: " + ", ".join(f"{c}={letters.count(c)}" for c in sorted(set(letters))))

    print("\n=== Null check: alpha = 0, AL minus BL (should be ~0; if not, the noise floor is larger than you think) ===")
    for arm, _ in arms:
        report_contrast(f"{arm}: E_AL - E_BL (alpha=0)", get(arm, 0.0, "E_AL") - get(arm, 0.0, "E_BL"))

    for key, label in [("E_AL", "PRIMARY: steady-state energy E_AL"), ("E_min", "SECONDARY: best energy found E_min")]:
        print(f"\n=== {label}  (negative = lower energy = better; paired across W0) ===")
        report_contrast("P0: partial(0) - full(0)  [reset effect]",
                        get("partial", 0.0, key) - get("full", 0.0, key))
        for a in args.alphas:
            Lf = get("full", a, key) - get("full", 0.0, key)
            Lp = get("partial", a, key) - get("partial", 0.0, key)
            print(f" alpha = {a:g}")
            report_contrast("L_full: learning effect, full resets", Lf)
            report_contrast("L_partial: learning effect, partial", Lp)
            report_contrast("Interaction = L_partial - L_full", Lp - Lf)

    print(f"\nSaved {args.out}_runs.csv (one row per run). Done in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
