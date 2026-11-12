"""
make_tables.py  --  Regenerate every table and CI reported in the README.

Reads raw simulation outputs:
    - so_repro_results.npz  (Figure 4 qualitative regimes and energies)
    - so_ablation_runs.csv  (2x2 ablation cell means, contrasts, and headroom analysis)
    - duty_default_runs.csv (Early-window duty cycle sweep)
    - duty_late_runs.csv    (Late-window duty cycle sweep)
    - duty_alpha_runs.csv   (Fixed-alpha duty cycle sweep)

Outputs formatted Markdown tables and paired contrasts with deterministic bootstrap 95% CIs
(seed = 0, n_boot = 10000).
"""
import numpy as np
import pandas as pd


def boot_ci(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])


def table_fig4():
    print("\n" + "=" * 80)
    print("TABLE 1: REPRODUCTION OF FIGURE 4 (from so_repro_results.npz)")
    print("=" * 80)
    data = np.load("so_repro_results.npz")
    alphas = data["alphas"]
    E = data["E"]
    dfro = data["dfro"]
    R = data["n_resets"]

    regime_names = [
        "A: No Convergence (not novel, not appropriate)",
        "B: Novel, Not Appropriate",
        "C: Appropriate, Not Novel",
        "D: Novel + Appropriate ('Self-Optimized' / Creative)",
    ]

    print(f"{'Regime':<45} | {'alpha':>10} | {'E_AL mean':>10} | {'Modal E':>10} | {'Modal share':>11} | {'||dW||_F':>10}")
    print("-" * 105)
    for i, a in enumerate(alphas):
        al = E[i, 2 * R:]
        vals, counts = np.unique(np.round(al, 6), return_counts=True)
        k = counts.argmax()
        mode_e = vals[k]
        frac = counts[k] / al.size
        dw = dfro[i, -1]
        print(f"{regime_names[i]:<45} | {a:10.1e} | {al.mean():10.2f} | {mode_e:10.2f} | {frac:11.2f} | {dw:10.2f}")


def table_ablation():
    print("\n" + "=" * 80)
    print("TABLE 2: 2x2 ABLATION CELL MEANS (from so_ablation_runs.csv)")
    print("=" * 80)
    df = pd.read_csv("so_ablation_runs.csv")

    def get(arm, a, key):
        out = []
        for w in range(10):
            sub = df[(df["w"] == w) & (df["reset"] == arm) & (np.isclose(df["alpha"], a))]
            out.append(sub[key].mean())
        return np.array(out)

    print(f"{'Reset':<8} | {'alpha':>10} | {'E_AL (mean +- SD)':>20} | {'E_min (mean +- SD)':>20} | {'Modal share':>11} | {'||dW||_F':>10}")
    print("-" * 90)
    for arm in ["full", "partial"]:
        for a in [0.0, 3e-7, 7e-7]:
            e = get(arm, a, "E_AL")
            m = get(arm, a, "E_min")
            f = get(arm, a, "AL_modal_frac")
            d = get(arm, a, "dW")
            print(f"{arm:<8} | {a:10.1e} | {e.mean():8.2f} +- {e.std(ddof=1):5.2f}     | {m.mean():8.2f} +- {m.std(ddof=1):5.2f}     | {f.mean():11.2f} | {d.mean():10.2f}")

    print("\nAblation Contrasts (Paired over W0, n=10):")
    p0 = get("partial", 0.0, "E_AL") - get("full", 0.0, "E_AL")
    ci_p0 = boot_ci(p0)
    print(f"  P0 (reset effect at alpha=0): {p0.mean():+.3f} [{ci_p0[0]:+.3f}, {ci_p0[1]:+.3f}]")

    for a in [3e-7, 7e-7]:
        Lf = get("full", a, "E_AL") - get("full", 0.0, "E_AL")
        Lp = get("partial", a, "E_AL") - get("partial", 0.0, "E_AL")
        I = Lp - Lf
        ci_Lf = boot_ci(Lf)
        ci_Lp = boot_ci(Lp)
        ci_I = boot_ci(I)
        print(f"  alpha={a:1.0e}: L_full={Lf.mean():+.3f} [{ci_Lf[0]:+.3f}, {ci_Lf[1]:+.3f}], L_partial={Lp.mean():+.3f} [{ci_Lp[0]:+.3f}, {ci_Lp[1]:+.3f}], Interaction={I.mean():+.3f} [{ci_I[0]:+.3f}, {ci_I[1]:+.3f}]")


def table_headroom():
    print("\n" + "=" * 80)
    print("TABLE 3: HEADROOM-NORMALISED ANALYSIS (from so_ablation_runs.csv)")
    print("=" * 80)
    df = pd.read_csv("so_ablation_runs.csv")
    agg = df.groupby(["w", "reset", "alpha"]).mean().reset_index()

    for a in [3e-7, 7e-7]:
        frac_f_list, frac_p_list, diff_list = [], [], []
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
        print(f"alpha = {a:1.0e}: Headroom full = {np.mean(frac_f_list)*100:.1f}%, partial = {np.mean(frac_p_list)*100:.1f}%, diff = {diffs.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")


def table_duty_early_vs_late():
    print("\n" + "=" * 80)
    print("TABLE 4: EARLY VS. LATE CONTRASTS (duty_default_runs.csv vs duty_late_runs.csv)")
    print("=" * 80)
    df_early = pd.read_csv("duty_default_runs.csv")
    df_late = pd.read_csv("duty_late_runs.csv")
    merged = pd.merge(df_early, df_late, on=["w", "run_seed", "T", "rho"], suffixes=("_early", "_late"))
    merged["diff"] = merged["E_AL_early"] - merged["E_AL_late"]

    print(f"{'T':>6} | {'rho':>7} | {'Early E_AL':>10} | {'Late E_AL':>10} | {'Diff (Early - Late)':>20} | {'Bootstrap 95% CI':>22}")
    print("-" * 85)
    for T in [500, 1000, 2000]:
        for rho in sorted(merged["rho"].unique()):
            sub = merged[(merged["T"] == T) & (np.isclose(merged["rho"], rho))]
            diffs = sub["diff"].values
            ci = boot_ci(diffs)
            print(f"{T:6d} | {rho:7.3f} | {sub['E_AL_early'].mean():10.2f} | {sub['E_AL_late'].mean():10.2f} | {diffs.mean():+20.3f} | [{ci[0]:+8.3f}, {ci[1]:+8.3f}]")


def table_duty_contrasts_and_flatness():
    print("\n" + "=" * 80)
    print("TABLE 5: DUTY CYCLE CONTRASTS & LATE FLATNESS")
    print("=" * 80)
    df_early = pd.read_csv("duty_default_runs.csv")
    df_late = pd.read_csv("duty_late_runs.csv")
    df_alpha = pd.read_csv("duty_alpha_runs.csv")

    print("1. Early window E_AL(1/e) - E_AL(1.0):")
    for T in [500, 1000, 2000]:
        sub = df_early[df_early["T"] == T]
        sub_1e = sub[np.isclose(sub["rho"], 0.36787944117144233)].sort_values("w")["E_AL"].values
        sub_10 = sub[np.isclose(sub["rho"], 1.0)].sort_values("w")["E_AL"].values
        d = sub_1e - sub_10
        ci = boot_ci(d)
        print(f"  T = {T:4d}: mean = {d.mean():+.2f}  [{ci[0]:+.2f}, {ci[1]:+.2f}]")

    print("\n2. Late window flatness and range across rho:")
    for T in [500, 1000, 2000]:
        sub = df_late[df_late["T"] == T]
        means = sub.groupby("rho")["E_AL"].mean()
        min_r, min_v = means.idxmin(), means.min()
        max_r, max_v = means.idxmax(), means.max()
        sub_1e = sub[np.isclose(sub["rho"], 0.36787944117144233)].sort_values("w")["E_AL"].values
        sub_10 = sub[np.isclose(sub["rho"], 1.0)].sort_values("w")["E_AL"].values
        d_10 = sub_1e - sub_10
        ci_10 = boot_ci(d_10)
        print(f"  T = {T:4d}: min mean = {min_v:.2f} (rho={min_r:.3f}), max mean = {max_v:.2f} (rho={max_r:.3f}), span = {max_v - min_v:.2f} units | contrast 1/e vs 1.0 = {d_10.mean():+.2f} [{ci_10[0]:+.2f}, {ci_10[1]:+.2f}]")

    print("\n3. Fixed-alpha overdose contrasts:")
    # T=2000 rho 0.5 vs 1.0
    sub_2000 = df_alpha[df_alpha["T"] == 2000]
    t2_05 = sub_2000[np.isclose(sub_2000["rho"], 0.5)].sort_values("w")["E_AL"].values
    t2_10 = sub_2000[np.isclose(sub_2000["rho"], 1.0)].sort_values("w")["E_AL"].values
    d2 = t2_05 - t2_10
    ci2 = boot_ci(d2)
    print(f"  T = 2000, rho 0.50 vs 1.00: mean = {d2.mean():+.2f}  [{ci2[0]:+.2f}, {ci2[1]:+.2f}]")

    # T=1000 rho 0.75 vs 1.0
    sub_1000 = df_alpha[df_alpha["T"] == 1000]
    t1_75 = sub_1000[np.isclose(sub_1000["rho"], 0.75)].sort_values("w")["E_AL"].values
    t1_10 = sub_1000[np.isclose(sub_1000["rho"], 1.0)].sort_values("w")["E_AL"].values
    d1 = t1_75 - t1_10
    ci1 = boot_ci(d1)
    print(f"  T = 1000, rho 0.75 vs 1.00: mean = {d1.mean():+.2f}  [{ci1[0]:+.2f}, {ci1[1]:+.2f}]")


def main():
    table_fig4()
    table_ablation()
    table_headroom()
    table_duty_early_vs_late()
    table_duty_contrasts_and_flatness()


if __name__ == "__main__":
    main()
