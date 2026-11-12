"""
analysis_headroom.py  --  Headroom-normalised analysis of the 2x2 ablation experiment.

Reads so_ablation_runs.csv and evaluates whether partial resets (25% Hamming flip)
exhibit synergy with Hebbian learning when normalized for baseline energy headroom.

Headroom definition:
    For each W0 instance w (w = 0..9) and reset arm (full or partial):
        Available headroom H = E_AL(arm, alpha=0) - E_min(arm, alpha=0)
    Fraction of headroom achieved by learning at learning rate alpha:
        frac(arm, alpha) = (E_AL(arm, alpha=0) - E_AL(arm, alpha)) / H
    Headroom interaction contrast:
        Delta_frac = frac(partial, alpha) - frac(full, alpha)
"""
import numpy as np
import pandas as pd


def boot_ci(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])


def main():
    df = pd.read_csv("so_ablation_runs.csv")
    agg = df.groupby(["w", "reset", "alpha"]).mean().reset_index()

    alphas = [3e-7, 7e-7]
    for a in alphas:
        frac_f_list = []
        frac_p_list = []
        diff_list = []
        for w in range(10):
            sub = agg[agg["w"] == w]
            f0 = sub[(sub["reset"] == "full") & (sub["alpha"] == 0)].iloc[0]
            p0 = sub[(sub["reset"] == "partial") & (sub["alpha"] == 0)].iloc[0]
            fa = sub[(sub["reset"] == "full") & (np.isclose(sub["alpha"], a))].iloc[0]
            pa = sub[(sub["reset"] == "partial") & (np.isclose(sub["alpha"], a))].iloc[0]

            Hf = f0["E_AL"] - f0["E_min"]
            Hp = p0["E_AL"] - p0["E_min"]

            frac_f = (f0["E_AL"] - fa["E_AL"]) / Hf
            frac_p = (p0["E_AL"] - pa["E_AL"]) / Hp

            frac_f_list.append(frac_f)
            frac_p_list.append(frac_p)
            diff_list.append(frac_p - frac_f)

        diffs = np.array(diff_list)
        ci = boot_ci(diffs)
        mean_f = np.mean(frac_f_list)
        mean_p = np.mean(frac_p_list)
        print(f"alpha = {a:g}:")
        print(f"  Headroom achieved: full = {mean_f*100:.1f}%, partial = {mean_p*100:.1f}%")
        print(f"  Difference (partial - full) = {diffs.mean():+.3f}  [95% CI: {ci[0]:+.3f}, {ci[1]:+.3f}]")
        print(f"  Units < 0: {(diffs < 0).sum()}/10")


if __name__ == "__main__":
    main()
