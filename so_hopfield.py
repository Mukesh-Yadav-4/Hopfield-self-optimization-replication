"""
so_hopfield.py  --  Self-Optimization (SO) Hopfield network, minimal reproduction harness.

Follows Weber, Guckelsberger & Froese (Artificial Life, 2025), Sec. 4 and the Fig. 4 protocol:

  * N = 100 bipolar nodes, modular W0 (Eq. 7): 20 modules of size k = 5,
    intra-module weights random +-1, inter-module weights random +-p (p = 0.1),
    symmetric, zero diagonal.
  * Asynchronous update (Eq. 1): pick one node uniformly at random, s_i = sign(sum_j W_L[i,j] s_j)
    (Heaviside: +1 for non-negative field).
  * Hebbian learning (Eq. 6): after EVERY update step, FULL matrix update
        W_L[i,j] += alpha * s_i * s_j   for all i != j     (diagonal stays 0, W_L stays symmetric)
  * T steps per reset, then the state is re-randomised; W_L is NOT reset.
  * Energy is always measured against the ORIGINAL W0 (Eq. 2), at the end of each reset.
  * Three phases of R resets each: before learning (BL, alpha = 0), learning (L), after learning (AL, W_L frozen).
  * The same run seed is used for every alpha, so the BL cloud is identical in every panel (as in the paper).

Run:
    python so_hopfield.py                                  # four default alphas, 1000 resets/phase
    python so_hopfield.py --resets 300 --alphas 5e-6       # quick look
    python so_hopfield.py --seed-w 3 --alphas 1e-7 3e-7 1e-6   # explore other alphas / W0 seeds

Notes:
  * The alpha -> regime mapping depends on the random W0 (the paper says so, Sec. 6.1). Your W0 differs from
    theirs, so match the four REGIMES, not the exact alphas or energies. If the default alphas don't give all
    four regimes for your seed, try other --seed-w values or shift alphas by factors of 2-3.
  * Reference scale from the paper's W0: BL mean = -127.2, sd = 7.0, best learned energy about -153.
  * No fastmath on purpose: keeps results bitwise reproducible across runs and exact symmetry of W_L.
"""
import argparse
import time

import numpy as np
from numba import njit


# --------------------------------------------------------------------------- problem construction
def make_W0(N=100, k=5, p=0.1, seed=0):
    """Modular weight matrix, paper Eq. (7): +-1 inside modules, +-p between modules, zero diagonal."""
    rng = np.random.default_rng(seed)
    module = np.arange(N) // k
    same = module[:, None] == module[None, :]
    sign = rng.choice([-1.0, 1.0], size=(N, N))
    sign = np.triu(sign, 1)
    sign = sign + sign.T                      # symmetric random signs, zero diagonal
    W0 = np.where(same, 1.0, p) * sign
    np.fill_diagonal(W0, 0.0)
    return W0


def assert_valid(W, name="W", atol=1e-12):
    assert np.all(np.isfinite(W)), f"{name}: non-finite entries"
    asym = np.max(np.abs(W - W.T))
    assert asym <= atol, f"{name}: not symmetric (max |W - W^T| = {asym:.3e})"
    assert np.all(np.diag(W) == 0.0), f"{name}: non-zero diagonal"


# --------------------------------------------------------------------------- numba kernels
@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)          # seeds numba's own RNG stream


@njit(cache=True)
def random_state(N):
    s = np.empty(N)
    for i in range(N):
        s[i] = 1.0 if np.random.random() < 0.5 else -1.0
    return s


@njit(cache=True)
def energy(W0, s):
    """E = -1/2 s^T W0 s  (Eq. 2 with no external input)."""
    N = s.shape[0]
    e = 0.0
    for i in range(N):
        acc = 0.0
        for j in range(N):
            acc += W0[i, j] * s[j]
        e += s[i] * acc
    return -0.5 * e


@njit(cache=True)
def async_step(W, s):
    """One asynchronous Hopfield update (Eq. 1) of a uniformly random node, using the CURRENT weights W."""
    N = s.shape[0]
    i = np.random.randint(0, N)
    h = 0.0
    for j in range(N):
        h += W[i, j] * s[j]
    s[i] = 1.0 if h >= 0.0 else -1.0


@njit(cache=True)
def hebb_update(W, s, alpha):
    """Full-matrix Hebbian step (Eq. 6): W += alpha * outer(s, s), diagonal kept at zero.
    s is +-1, so alpha*s[a]*s[b] is exact and W[a,b], W[b,a] receive bitwise-identical increments."""
    N = s.shape[0]
    for a in range(N):
        sa = alpha * s[a]
        for b in range(N):
            W[a, b] += sa * s[b]
        W[a, a] = 0.0


@njit(cache=True)
def run_resets(W_L, W0, n_resets, T, alpha, E_out, dfro_out, dmax_out, offset):
    """Run n_resets resets in place on W_L. Records, after each reset:
       E_out    : energy of the final state w.r.t. W0
       dfro_out : ||W_L - W0||_F
       dmax_out : max |W_L - W0|
    alpha == 0 means no learning (BL / AL phases)."""
    N = W_L.shape[0]
    learn = alpha != 0.0
    for r in range(n_resets):
        s = random_state(N)
        for t in range(T):
            async_step(W_L, s)
            if learn:
                hebb_update(W_L, s, alpha)
        E_out[offset + r] = energy(W0, s)
        fro = 0.0
        mx = 0.0
        for a in range(N):
            for b in range(N):
                d = W_L[a, b] - W0[a, b]
                fro += d * d
                ad = abs(d)
                if ad > mx:
                    mx = ad
        dfro_out[offset + r] = np.sqrt(fro)
        dmax_out[offset + r] = mx


@njit(cache=True)
def relax_trace(W, W0, T):
    """Energy w.r.t. W0 at every step of one relaxation from a random state (no learning)."""
    N = W.shape[0]
    s = random_state(N)
    out = np.empty(T + 1)
    out[0] = energy(W0, s)
    for t in range(T):
        async_step(W, s)
        out[t + 1] = energy(W0, s)
    return out


# --------------------------------------------------------------------------- protocol
def run_protocol(W0, alpha, n_resets=1000, T=1000, seed=1):
    """BL (alpha=0) -> L (alpha) -> AL (frozen W_L). Returns energies and ||dW|| traces of length 3*n_resets."""
    _seed(seed)                                   # identical dynamics stream for every alpha
    W_L = W0.copy()
    n = 3 * n_resets
    E = np.empty(n)
    dfro = np.empty(n)
    dmax = np.empty(n)
    run_resets(W_L, W0, n_resets, T, 0.0, E, dfro, dmax, 0)
    run_resets(W_L, W0, n_resets, T, alpha, E, dfro, dmax, n_resets)
    run_resets(W_L, W0, n_resets, T, 0.0, E, dfro, dmax, 2 * n_resets)
    assert_valid(W_L, "W_L after protocol")
    return E, dfro, dmax, W_L


def classify(E, n_resets, conv_frac=0.9):
    """Heuristic stand-in for the paper's four regimes, using the AL phase vs the BL distribution.
       converged  : >= conv_frac of AL energies sit on one value
       appropriate: converged AND that energy is below the BL mean
       novel      : converged energy lies outside the observed BL range [min, max]"""
    BL = E[:n_resets]
    AL = E[2 * n_resets:]
    mu = BL.mean()
    vals, counts = np.unique(np.round(AL, 6), return_counts=True)
    k = counts.argmax()
    mode_E, frac = float(vals[k]), counts[k] / AL.size
    if frac < conv_frac:
        label = "A: no convergence (not novel, not appropriate)"
    else:
        below = mode_E < mu
        outside = (mode_E < BL.min()) or (mode_E > BL.max())
        if below and outside:
            label = "D: novel + appropriate (creative)"
        elif below:
            label = "C: appropriate, not novel"
        elif outside:
            label = "B: novel, not appropriate"
        else:
            label = "?: converged inside BL range, above mean"
    return dict(label=label, mode_E=mode_E, mode_frac=float(frac),
                BL_mean=float(mu), BL_sd=float(BL.std()), BL_min=float(BL.min()), BL_max=float(BL.max()))


# --------------------------------------------------------------------------- self tests
def selftest(W0):
    print("[selftest] running ...")
    N = W0.shape[0]

    # 1) W0 structure
    assert_valid(W0, "W0")
    k = 5
    module = np.arange(N) // k
    same = module[:, None] == module[None, :]
    off = ~np.eye(N, dtype=bool)
    assert np.all(np.abs(W0[same & off]) == 1.0), "intra-module weights must be +-1"
    assert np.allclose(np.abs(W0[~same]), np.abs(W0[~same])[0]), "inter-module weights must share one magnitude"

    # 2) hebb_update == NumPy reference
    rng = np.random.default_rng(0)
    A = rng.normal(size=(12, 12))
    W = (A + A.T) / 2
    np.fill_diagonal(W, 0.0)
    s = rng.choice([-1.0, 1.0], 12)
    ref = W + 0.37 * np.outer(s, s)
    np.fill_diagonal(ref, 0.0)
    W2 = W.copy()
    hebb_update(W2, s, 0.37)
    assert np.allclose(W2, ref, atol=1e-14), "hebb_update disagrees with NumPy outer-product reference"
    assert_valid(W2, "W after hebb_update")

    # 3) energy is non-increasing under async updates when W_L == W0 (no learning)
    _seed(123)
    for _ in range(5):
        tr = relax_trace(W0.copy(), W0, 1000)
        assert np.max(np.diff(tr)) <= 1e-9, "energy increased during relaxation: update rule/W0 is wrong"

    # 4) learning bookkeeping on a short protocol
    R, T, a = 15, 1000, 1e-5
    E, dfro, dmax, W_L = run_protocol(W0, a, n_resets=R, T=T, seed=7)
    assert np.all(dfro[:R] == 0.0), "BL phase must leave W_L == W0"
    assert dfro[2 * R - 1] > 0.0, "learning phase did not change W_L (learning inactive!)"
    assert np.ptp(dfro[2 * R:]) == 0.0, "AL phase must freeze W_L"
    assert dmax.max() <= a * T * R + 1e-12, "|dW| exceeds the theoretical per-entry bound"
    E0, dfro0, _, _ = run_protocol(W0, 0.0, n_resets=R, T=T, seed=7)
    assert np.all(dfro0 == 0.0), "alpha = 0 must never change W_L"
    print(f"[selftest] OK  (learning moved W_L by ||dW||_F = {dfro[2 * R - 1]:.4f} after {R} learning resets at alpha={a:g})")


# --------------------------------------------------------------------------- plotting
def plot_results(results, n_resets, W0_seed, path_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    ncols = 2
    nrows = (n + 1) // 2
    xs = np.arange(3 * n_resets)
    cols = ["#08306b"] * n_resets + ["#b2182b"] * n_resets + ["#4fc3f7"] * n_resets
    lo = min(r["E"].min() for r in results) - 3
    hi = max(r["E"].max() for r in results) + 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        ax.scatter(xs, r["E"], s=1.5, c=cols, linewidths=0)
        ax.set_ylim(lo, hi)
        ax.set_title(f"alpha = {r['alpha']:g}\n{r['info']['label']}", fontsize=9)
        ax.set_xlabel("Resets")
        ax.set_ylabel("Energy (w.r.t. W0)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Fig. 4 reproduction (blue: BL, red: learning, light blue: AL)  |  W0 seed {W0_seed}")
    fig.tight_layout()
    fig.savefig(f"{path_prefix}_fig4.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in results:
        ax.plot(xs, np.maximum(r["dfro"], 1e-12), label=f"alpha={r['alpha']:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Resets")
    ax.set_ylabel("||W_L - W0||_F")
    ax.legend(fontsize=8)
    ax.set_title("Weight change (diagnostic: must be 0 in BL, grow in L, flat in AL)")
    fig.tight_layout()
    fig.savefig(f"{path_prefix}_dW.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--seed-w", type=int, default=0, help="seed for W0")
    ap.add_argument("--seed-run", type=int, default=1, help="seed for states / update order")
    ap.add_argument("--resets", type=int, default=1000, help="resets per phase (BL / L / AL)")
    ap.add_argument("--steps", type=int, default=1000, help="asynchronous steps per reset (T)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[3e-8, 5e-5, 3e-6, 5e-7],
                    help="default order = paper panels (a)-(d)")
    ap.add_argument("--out", type=str, default="so_repro")
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()

    W0 = make_W0(args.N, args.k, args.p, args.seed_w)
    if not args.skip_selftest:
        selftest(W0)
    if args.selftest_only:
        return

    results = []
    print(f"\nN={args.N}  k={args.k}  p={args.p}  T={args.steps}  resets/phase={args.resets}  "
          f"seed_w={args.seed_w}  seed_run={args.seed_run}")
    for alpha in args.alphas:
        t0 = time.time()
        E, dfro, dmax, W_L = run_protocol(W0, alpha, args.resets, args.steps, args.seed_run)
        info = classify(E, args.resets)
        results.append(dict(alpha=alpha, E=E, dfro=dfro, dmax=dmax, info=info))
        print(f"\nalpha = {alpha:g}   ({time.time() - t0:.1f}s)")
        print(f"  BL: mean {info['BL_mean']:.2f}  sd {info['BL_sd']:.2f}  range [{info['BL_min']:.2f}, {info['BL_max']:.2f}]")
        print(f"  AL: modal energy {info['mode_E']:.2f}  (share of AL resets on it: {info['mode_frac']:.2f})")
        print(f"  ||W_L - W0||_F = {dfro[-1]:.4f}   max|dW| = {dmax[-1]:.4f}")
        print(f"  -> {info['label']}")

    plot_results(results, args.resets, args.seed_w, args.out)
    np.savez(f"{args.out}_results.npz", W0=W0, alphas=np.array(args.alphas),
             E=np.stack([r["E"] for r in results]), dfro=np.stack([r["dfro"] for r in results]),
             dmax=np.stack([r["dmax"] for r in results]), n_resets=args.resets, T=args.steps)
    print(f"\nSaved {args.out}_fig4.png, {args.out}_dW.png, {args.out}_results.npz")
    print("Expected (paper Fig. 4): a flat cloud (no convergence); a jump to ONE state ABOVE the BL range; "
          "convergence to one state INSIDE the BL range but below its mean; a gradual descent to ONE state BELOW the BL range.")


if __name__ == "__main__":
    main()
