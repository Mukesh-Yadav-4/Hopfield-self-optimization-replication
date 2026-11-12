"""
so_duty_cycle.py  --  Does a ~1/e "duty cycle" matter in the Self-Optimization (SO) Hopfield model?

Why this design (and not "RR intervals as reset intervals"):
    Froese's 1/e framework (as summarised on his OIST page) concerns the FRACTION of a work-constraint cycle over
    which effort is allocated (~37%), with the cardiac duty cycle as one empirical support. That is a duty-cycle
    claim, not a claim about interval variability. So the faithful SO analogue is:

        one reset period = one cycle of T asynchronous steps
        Hebbian plasticity ("effort") is ON for the first rho*T steps of every cycle, OFF for the rest
        rho = 1/e ~ 0.368 is one labelled point on a sweep of rho

Dose control (the crux):
    rho changes how much learning happens unless you hold it fixed. --dose-rule cycle (default) fixes the total Hebbian
    dose per cycle, D = alpha_eff * plastic_steps  (so alpha_eff = D / (rho*T)): only WHEN/how concentrated the dose
    is changes. --dose-rule alpha keeps alpha fixed (dose then grows with rho).

Cross-scale control (the part that makes it a test of a FRACTION, not just of convergence time):
    Run several cycle lengths T. A duty-cycle (fraction) effect should sit at the same rho for every T. An effect that
    tracks convergence time should sit at the same number of plastic STEPS (rho*T) instead. The second plot shows both axes.

Pre-stated expectation (so a null is an acceptable outcome):
    The model has no built-in cycle scale. Async Hopfield relaxation settles in roughly O(100) steps, so plasticity
    that only covers the first few tens of steps imprints transients, while larger windows imprint the settled attractor.
    I therefore expect (i) a knee at small rho whose position scales like 1/T (absolute steps), (ii) a flat plateau
    above it, and NO preference for rho = 1/e. A 1/e optimum that is T-invariant would be a surprise needing more controls.

Protocol per run = so_hopfield's BL (alpha=0) / L (duty-gated alpha_eff) / AL (frozen) with full random resets.
Outcome E_AL = mean energy (w.r.t. W0) over the AL phase; lower is better. Unit of analysis = W0 instance (paired).

Run:
    python so_duty_cycle.py --selftest-only
    python so_duty_cycle.py                                 # 10 W0 x 1 run seed, T in {500,1000,2000}, ~11 min
    python so_duty_cycle.py --w-seeds 4 --resets 300 --Ts 500 1000      # quick look
    python so_duty_cycle.py --dose-rule alpha               # fixed alpha instead of fixed dose
"""
import argparse
import csv
import time

import numpy as np
from numba import njit

from so_ablation import boot_ci, report_contrast
from so_hopfield import (_seed, assert_valid, async_step, energy, hebb_update, make_W0, random_state,
                         run_protocol)

INV_E = float(np.exp(-1.0))
RHO_GRID = [0.05, 0.1, 0.2, 0.3, INV_E, 0.5, 0.75, 1.0]


@njit(cache=True)
def run_resets_duty(W_L, W0, n_resets, T, alpha, plastic_steps, late, E_out, dfro_out, offset):
    """Full random resets; Hebbian plasticity (alpha) only during `plastic_steps` steps of each T-step cycle:
    the first ones (late = False, 'work then rest') or the last ones (late = True)."""
    N = W_L.shape[0]
    learn = alpha != 0.0
    start = T - plastic_steps if late else 0
    stop = T if late else plastic_steps
    for r in range(n_resets):
        s = random_state(N)
        for t in range(T):
            async_step(W_L, s)
            if learn and t >= start and t < stop:
                hebb_update(W_L, s, alpha)
        E_out[offset + r] = energy(W0, s)
        fro = 0.0
        for a in range(N):
            for b in range(N):
                d = W_L[a, b] - W0[a, b]
                fro += d * d
        dfro_out[offset + r] = np.sqrt(fro)


def run_protocol_duty(W0, alpha, plastic_steps, T, n_resets, seed, late=False):
    _seed(seed)
    W_L = W0.copy()
    E = np.empty(3 * n_resets)
    dfro = np.empty(3 * n_resets)
    run_resets_duty(W_L, W0, n_resets, T, 0.0, plastic_steps, late, E, dfro, 0)
    run_resets_duty(W_L, W0, n_resets, T, alpha, plastic_steps, late, E, dfro, n_resets)
    run_resets_duty(W_L, W0, n_resets, T, 0.0, plastic_steps, late, E, dfro, 2 * n_resets)
    assert_valid(W_L, "W_L after duty-cycle protocol")
    return E, dfro


def selftest():
    print("[selftest] running ...")
    W0 = make_W0(100, 5, 0.1, 0)
    R, T, a, seed = 20, 300, 1e-5, 3
    # 1) plastic_steps = T reproduces the verified so_hopfield protocol bit for bit
    E_ref, dfro_ref, _, _ = run_protocol(W0, a, R, T, seed)
    E_new, dfro_new = run_protocol_duty(W0, a, T, T, R, seed)
    assert np.array_equal(E_ref, E_new), "duty protocol with rho=1 deviates from verified so_hopfield protocol"
    # 2) no plastic steps -> no learning
    _, d0 = run_protocol_duty(W0, a, 0, T, R, seed)
    assert np.all(d0 == 0.0), "plastic_steps = 0 changed W_L"
    # 3) at fixed alpha, a shorter plastic window changes W_L less; AL frozen; BL untouched
    _, d_small = run_protocol_duty(W0, a, T // 5, T, R, seed)
    assert d_small[2 * R - 1] < dfro_new[2 * R - 1], "smaller plastic window should move W_L less at fixed alpha"
    assert np.ptp(d_small[2 * R:]) == 0.0 and np.all(d_small[:R] == 0.0), "phase bookkeeping broken"
    # 4) early vs late window both run and differ
    E_e, _ = run_protocol_duty(W0, a, T // 5, T, R, seed, late=False)
    E_l, _ = run_protocol_duty(W0, a, T // 5, T, R, seed, late=True)
    assert not np.array_equal(E_e, E_l), "early/late windows should differ"
    print("[selftest] OK  (rho=1 identical to so_hopfield; rho=0 inert; window bookkeeping ok)")


def analyse(args, rows, Ts, rhos):
    W = list(range(args.w_seeds))

    def curve(T, rho, key="E_AL"):
        return np.array([np.mean([r[key] for r in rows if r["w"] == w and r["T"] == T and abs(r["rho"] - rho) < 1e-9])
                         for w in W])

    print("\n=== E_AL by rho and cycle length T (mean over W0; lower is better) ===")
    head = f"{'rho':>7s}" + "".join(f"{'T=' + str(T) + ' (steps)':>22s}" for T in Ts)
    print(head)
    for rho in rhos:
        cells = []
        for T in Ts:
            c = curve(T, rho)
            cells.append(f"{c.mean():>11.2f} ({max(1, int(round(rho * T))):>4d})")
        mark = "  <- 1/e" if abs(rho - INV_E) < 1e-9 else ""
        print(f"{rho:>7.3f}" + "".join(f"{x:>22s}" for x in cells) + mark)

    rng = np.random.default_rng(0)
    for T in Ts:
        print(f"\n=== T = {T}: paired contrasts (negative = lower energy; n = {len(W)} W0 instances) ===")
        e_inv = curve(T, INV_E)
        report_contrast("E_AL(1/e) - E_AL(1.0)", e_inv - curve(T, 1.0))
        for nb in (0.3, 0.5):
            if any(abs(r - nb) < 1e-9 for r in rhos):
                report_contrast(f"E_AL(1/e) - E_AL({nb})", e_inv - curve(T, nb))
        # which rho minimises the mean curve, and how stable is that choice across W0 resamples?
        M = np.stack([curve(T, rho) for rho in rhos])            # (n_rho, n_w)
        idx = rng.integers(0, len(W), size=(5000, len(W)))
        am = np.array([np.argmin(M[:, i].mean(axis=1)) for i in idx])
        share = ", ".join(f"{rho:.2f}: {np.mean(am == j):.2f}" for j, rho in enumerate(rhos) if np.mean(am == j) > 0)
        print(f"  bootstrap argmin(rho) share -> {share}")
        floor = M.mean(axis=1).min()
        print("  plastic steps needed to get within 1.0 energy unit of the best rho: "
              + ", ".join(f"{rho:.2f}={max(1, int(round(rho * T)))}" for j, rho in enumerate(rhos)
                          if M[j].mean() <= floor + 1.0)[:120])
    return curve


def plot(args, curve, Ts, rhos, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W = args.w_seeds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for T in Ts:
        mean, lo, hi = [], [], []
        for rho in rhos:
            c = curve(T, rho)
            mean.append(c.mean())
            ci = boot_ci(c) if W > 1 else (c.mean(), c.mean())
            lo.append(ci[0]); hi.append(ci[1])
        steps = [max(1, int(round(r * T))) for r in rhos]
        for ax, x in ((axes[0], rhos), (axes[1], steps)):
            ax.plot(x, mean, marker="o", ms=3, label=f"T={T}")
            ax.fill_between(x, lo, hi, alpha=0.15)
    axes[0].axvline(INV_E, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("duty cycle rho = plastic fraction of each cycle")
    axes[1].set_xlabel("plastic steps per cycle (rho * T)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_ylabel("E_AL (lower = better)")
        ax.legend(fontsize=8)
    axes[0].set_title("A fraction effect would collapse here, at the same rho for every T", fontsize=9)
    axes[1].set_title("A convergence-time effect would collapse here, at the same step count", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dose", type=float, default=7e-4, help="Hebbian dose per cycle D (7e-4 = alpha 7e-7 at T=1000)")
    ap.add_argument("--dose-rule", choices=["cycle", "alpha"], default="cycle")
    ap.add_argument("--Ts", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--rhos", type=float, nargs="+", default=RHO_GRID)
    ap.add_argument("--late", action="store_true", help="plasticity in the LAST rho*T steps instead of the first")
    ap.add_argument("--w-seeds", type=int, default=10)
    ap.add_argument("--run-seeds", type=int, default=1)
    ap.add_argument("--resets", type=int, default=1000, help="resets per phase")
    ap.add_argument("--out", type=str, default="so_duty")
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()

    if not args.skip_selftest:
        selftest()
    if args.selftest_only:
        return

    rhos = sorted(set(args.rhos))
    rows = []
    t0 = time.time()
    print(f"\nrho grid {', '.join(f'{r:.3f}' for r in rhos)}; T in {args.Ts}; dose rule '{args.dose_rule}', "
          f"D={args.dose:g}; window {'late' if args.late else 'early'}; {args.w_seeds} W0 x {args.run_seeds} run seeds")
    for w in range(args.w_seeds):
        W0 = make_W0(100, 5, 0.1, w)
        for rs in range(args.run_seeds):
            for T in args.Ts:
                for rho in rhos:
                    plastic = max(1, int(round(rho * T)))
                    alpha = args.dose / plastic if args.dose_rule == "cycle" else args.dose / 1000.0
                    E, dfro = run_protocol_duty(W0, alpha, plastic, T, args.resets, 1 + rs, args.late)
                    AL = E[2 * args.resets:]
                    vals, counts = np.unique(np.round(AL, 6), return_counts=True)
                    rows.append(dict(w=w, run_seed=1 + rs, T=T, rho=rho, plastic_steps=plastic, alpha_eff=alpha,
                                     E_BL=float(E[:args.resets].mean()), E_AL=float(AL.mean()),
                                     AL_modal_frac=float(counts.max() / AL.size), dW=float(dfro[-1])))
        print(f"  W0 {w} done  ({time.time() - t0:.0f}s elapsed)")

    with open(f"{args.out}_runs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    curve = analyse(args, rows, args.Ts, rhos)
    plot(args, curve, args.Ts, rhos, f"{args.out}_curves.png")
    print(f"\nSaved {args.out}_runs.csv and {args.out}_curves.png  ({time.time() - t0:.0f}s total)")
    print("Read the plot first: do the T-curves line up on the left (fraction) or on the right (absolute steps)?")


if __name__ == "__main__":
    main()
