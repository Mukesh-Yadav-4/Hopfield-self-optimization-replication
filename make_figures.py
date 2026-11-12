"""
make_figures.py  --  Generate ablation_E_AL.png and early_vs_late.png from existing CSVs.

Uses seed = 0 and 10,000 bootstrap resamples for all confidence intervals.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def boot_ci(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])


def make_ablation_figure():
    df = pd.read_csv("so_ablation_runs.csv")
    agg = df.groupby(["w", "reset", "alpha"])["E_AL"].mean().reset_index()

    conditions = [
        ("full", 0.0, "full\nalpha=0"),
        ("full", 3e-7, "full\n3e-7"),
        ("full", 7e-7, "full\n7e-7"),
        ("partial", 0.0, "partial\nalpha=0"),
        ("partial", 3e-7, "partial\n3e-7"),
        ("partial", 7e-7, "partial\n7e-7"),
    ]

    xs = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)

    # Matrix of shape (10, 6) for W0 instances across conditions
    w_matrix = np.zeros((10, len(conditions)))
    cond_means = []

    for j, (arm, a, _) in enumerate(conditions):
        for w in range(10):
            val = agg[(agg["w"] == w) & (agg["reset"] == arm) & (np.isclose(agg["alpha"], a))]["E_AL"].values[0]
            w_matrix[w, j] = val
        cond_means.append(np.mean(w_matrix[:, j]))

    # Draw thin grey line per W0 instance
    for w in range(10):
        ax.plot(xs, w_matrix[w, :], color="#888888", alpha=0.55, lw=1.1, zorder=1)

    # Bold marker and line for the mean
    ax.plot(xs, cond_means, color="#08306b", lw=2.4, marker="o", ms=7, zorder=3, label="Condition Mean")

    # Annotate mean values above/below points
    for j, m in enumerate(cond_means):
        ax.annotate(f"{m:.2f}", (xs[j], m), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5, fontweight="bold", color="#08306b")

    # Add vertical separator between full and partial resets
    ax.axvline(2.5, color="#bbbbbb", ls="--", lw=1.0, zorder=0)
    ax.text(1.0, -125.0, "Full Resets (100%)", ha="center", fontsize=9.5, fontweight="bold", color="#333333")
    ax.text(4.0, -125.0, "Partial Resets (25%)", ha="center", fontsize=9.5, fontweight="bold", color="#333333")

    ax.set_xticks(xs)
    ax.set_xticklabels([c[2] for c in conditions], fontsize=9)
    ax.set_ylabel("Mean E_AL (energy units, w.r.t. W0; lower is better)", fontsize=9.5)
    ax.set_xlabel("Condition (Reset type and learning rate alpha)", fontsize=9.5)
    ax.set_title("Steady-state energy by reset type and learning rate (n = 10 W0 instances; lower is better)", fontsize=10.5, pad=12)
    ax.grid(True, ls=":", alpha=0.5, zorder=0)
    ax.set_ylim(-160, -122)

    fig.tight_layout()
    fig.savefig("ablation_E_AL.png", dpi=150)
    plt.close(fig)
    print("Saved ablation_E_AL.png")


def make_early_vs_late_figure():
    df_early = pd.read_csv("duty_default_runs.csv")
    df_late = pd.read_csv("duty_late_runs.csv")
    merged = pd.merge(df_early, df_late, on=["w", "run_seed", "T", "rho"], suffixes=("_early", "_late"))
    merged["diff"] = merged["E_AL_early"] - merged["E_AL_late"]

    rhos = sorted(merged["rho"].unique())
    Ts = [500, 1000, 2000]
    colors = {500: "#d95f02", 1000: "#7570b3", 2000: "#1b9e77"}

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)

    for T in Ts:
        diff_means = []
        cis_lo = []
        cis_hi = []
        for rho in rhos:
            sub = merged[(merged["T"] == T) & (np.isclose(merged["rho"], rho))]
            diffs = sub["diff"].values
            ci = boot_ci(diffs, n_boot=10000, seed=0)
            diff_means.append(diffs.mean())
            cis_lo.append(ci[0])
            cis_hi.append(ci[1])

        ax.plot(rhos, diff_means, color=colors[T], lw=2.0, marker="o", ms=4, label=f"T = {T}")
        ax.fill_between(rhos, cis_lo, cis_hi, color=colors[T], alpha=0.18)

    # Horizontal zero line
    ax.axhline(0, color="k", ls="--", lw=1.0, zorder=2, label="Equal Performance (0)")
    # Mark 1/e
    INV_E = 1.0 / np.e
    ax.axvline(INV_E, color="#666666", ls=":", lw=1.0, zorder=2)
    ax.text(INV_E + 0.015, 14.5, "rho = 1/e ~ 0.368", fontsize=8.5, color="#444444")

    ax.set_xlabel("Duty cycle fraction rho", fontsize=9.5)
    ax.set_ylabel("E_AL(early) - E_AL(late) (energy units)", fontsize=9.5)
    ax.set_title("Early minus late plasticity window at equal dose (positive = late is better)", fontsize=10.5, pad=12)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, ls=":", alpha=0.5, zorder=0)
    ax.set_ylim(-3, 19)

    fig.tight_layout()
    fig.savefig("early_vs_late.png", dpi=150)
    plt.close(fig)
    print("Saved early_vs_late.png")


def main():
    make_ablation_figure()
    make_early_vs_late_figure()


if __name__ == "__main__":
    main()
