"""
Automated Analysis and Paper Artifact Generator for Experiment 3 (E3) and Experiment 8 (E8).
Generates:
1. Figure 3: Consortium Scaling Curves (Panel a: PRR vs alpha across N={2,3,4}; Panel b: Uniform Dilution Slice)
2. Figure 5: Preference vs. Mean Capability Pareto Frontier
3. Figure 6: Mechanistic Predictors of Alignment Failure (Scatter + Spearman rank correlation rho and p-values)
4. Table 3: Consortium Merging Master Table (LaTeX and Markdown)
Copies all generated figures directly into the artifact directory for inline UI preview.
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import shutil
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACT_DIR = r"C:\Users\vedan\.gemini\antigravity\brain\c4af8d62-e4c1-481f-bd8a-d208a5ae2a79"

# Visual styling
plt.rcParams["font.family"] = "sans-serif"
COLORS = {
    "linear": "#1f77b4",
    "ties": "#2ca02c",
    "dare_ties": "#d62728"
}
LABELS = {
    "linear": "Linear Task Arithmetic",
    "ties": "TIES-Merging",
    "dare_ties": "DARE-TIES"
}
MARKERS_BY_N = {
    2: "o",
    3: "s",
    4: "D"
}


def load_e2_data(e2_path="outputs/results/e2_sweep_results.json") -> dict:
    """Loads Experiment 2 data (N=2) and reformats into unified structure."""
    if not os.path.exists(e2_path):
        print(f"Warning: E2 results not found at {e2_path}")
        return {}

    with open(e2_path, "r") as f:
        data = json.load(f)

    reformatted = {}
    for k, v in data.items():
        # E2 keys are e.g. "linear_0.5"
        parts = k.split("_")
        method = "_".join(parts[:-1])
        if method not in ["linear", "ties", "dare_ties"]:
            continue
        try:
            alpha = float(parts[-1])
        except ValueError:
            continue

        prr = v.get("prr", 0.0)
        cr_math = v.get("cr_math", 0.0)
        # In N=2, only math was merged with preference
        mean_cr = cr_math

        reformatted[f"N2_{method}_{alpha:.4f}"] = {
            "N": 2,
            "method": method,
            "alpha_pref": alpha,
            "prr": prr,
            "mean_cr": mean_cr,
            "cr_math": cr_math,
            "svamp_acc": v.get("svamp_acc", 0.0),
            "dpo_margin": v.get("dpo_margin", 0.0)
        }
    return reformatted


def plot_figure3(unified_data: dict, output_dir: str = "outputs/results"):
    """
    Figure 3: Consortium Scaling & Dilution.
    Panel (a): 3 Subplots (Linear, TIES, DARE-TIES) showing PRR vs alpha_pref across N=2, 3, 4.
    Panel (b): Uniform Dilution Slice (PRR and Mean CR at alpha = 1/N).
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), dpi=300)

    methods = ["linear", "ties", "dare_ties"]
    n_colors = {2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c"}

    # Panels 1-3: PRR vs alpha_pref per algorithm across N
    for idx, method in enumerate(methods):
        ax = axes[idx]
        for N in [2, 3, 4]:
            pts = []
            for k, row in unified_data.items():
                if row["N"] == N and row["method"] == method:
                    pts.append((row["alpha_pref"], row["prr"], row.get("prr_ci95", 0.0)))
            if pts:
                pts.sort(key=lambda x: x[0])
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                yerrs = np.array([p[2] for p in pts])
                ax.plot(
                    xs, ys,
                    marker=MARKERS_BY_N[N],
                    linewidth=2.2,
                    markersize=6,
                    label=f"$N={N}$ ({'Math' if N==2 else 'Math+Code' if N==3 else 'Math+Code+Instr'})",
                    color=n_colors[N]
                )
                if np.any(yerrs > 0):
                    ax.fill_between(xs, ys - yerrs, ys + yerrs, color=n_colors[N], alpha=0.15)

        ax.axhline(100.0, color="purple", linestyle=":", linewidth=1.5, alpha=0.7, label=r"$E_{\text{pref}}$ Anchor (100%)")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.6, label=r"Base $B_0$ (0%)")
        ax.set_title(f"({chr(97 + idx)}) {LABELS[method]}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(r"Preference Weight $\alpha_{\text{pref}}$", fontsize=11)
        ax.set_ylabel(r"Preference Retention Ratio PRR (%)" if idx == 0 else "", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9, loc="best")
        ax.set_xlim(0.05, 0.95)

    # Panel 4: Uniform Dilution Slice (alpha = 1/N)
    ax4 = axes[3]
    uniform_alphas = {2: 0.50, 3: 0.3333, 4: 0.2500}
    bar_width = 0.25
    x_indices = np.arange(3)  # N = 2, 3, 4

    for m_idx, method in enumerate(methods):
        prr_vals = []
        for N in [2, 3, 4]:
            target_alpha = uniform_alphas[N]
            # Find closest entry
            best_match = None
            min_dist = 1e9
            for k, row in unified_data.items():
                if row["N"] == N and row["method"] == method:
                    dist = abs(row["alpha_pref"] - target_alpha)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = row
            prr_vals.append(best_match["prr"] if best_match else 0.0)

        offset = (m_idx - 1) * bar_width
        bars = ax4.bar(
            x_indices + offset,
            prr_vals,
            width=bar_width,
            label=LABELS[method],
            color=COLORS[method],
            alpha=0.88,
            edgecolor="black",
            linewidth=0.8
        )
        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1.5 if height >= 0 else height - 5.0,
                f"{height:.1f}%",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=8, fontweight="bold"
            )

    ax4.axhline(0.0, color="gray", linestyle="--", linewidth=1.2)
    ax4.set_title("(d) Uniform Dilution Slice ($w_k = 1/N$)", fontsize=12, fontweight="bold", pad=8)
    ax4.set_xlabel("Consortium Size $N$", fontsize=11)
    ax4.set_ylabel("PRR (%) at Equal Weight", fontsize=11)
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels(["$N=2$\n($w=0.50$)", "$N=3$\n($w=0.33$)", "$N=4$\n($w=0.25$)"])
    ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax4.legend(frameon=True, fontsize=8.5, loc="best")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure3_consortium_scaling.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"📊 Saved Figure 3 to: {fig_path}")

    # Copy to artifacts directory
    if os.path.exists(ARTIFACT_DIR):
        shutil.copy(fig_path, os.path.join(ARTIFACT_DIR, "figure3_consortium_scaling.png"))


def plot_figure5(unified_data: dict, output_dir: str = "outputs/results"):
    """
    Figure 5: Preference vs. Capability Pareto Frontier.
    Scatter plot: PRR (%) vs Mean CR (%) across all merges, highlighting Pareto boundary.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)

    for method in ["linear", "ties", "dare_ties"]:
        for N in [2, 3, 4]:
            pts = []
            for k, row in unified_data.items():
                if row["N"] == N and row["method"] == method:
                    pts.append((row["prr"], row["mean_cr"], row["alpha_pref"]))
            if pts:
                xs, ys, alphas = zip(*pts)
                ax.scatter(
                    xs, ys,
                    color=COLORS[method],
                    marker=MARKERS_BY_N[N],
                    s=65,
                    alpha=0.85,
                    label=f"{LABELS[method]} ($N={N}$)"
                )

    # Specialists & Base Anchors
    ax.scatter([100.0], [-6.2], color="purple", marker="*", s=220, zorder=5, label=r"$E_{\text{pref}}$ Anchor")
    ax.scatter([0.0], [100.0], color="green", marker="*", s=220, zorder=5, label=r"Specialists ($100\%$ CR)")
    ax.scatter([0.0], [0.0], color="black", marker="X", s=140, zorder=5, label=r"Base $B_0$ ($0\%$ Anchor)")

    ax.set_title("Figure 5: Preference vs. Mean Capability Pareto Frontier", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(r"Preference Retention Ratio PRR (%)", fontsize=11, fontweight="medium")
    ax.set_ylabel(r"Mean Capability Retention $\overline{\text{CR}}$ (%)", fontsize=11, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend(frameon=True, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.02, 0.02), ncol=2)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure5_pareto_frontier.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"📊 Saved Figure 5 to: {fig_path}")

    if os.path.exists(ARTIFACT_DIR):
        shutil.copy(fig_path, os.path.join(ARTIFACT_DIR, "figure5_pareto_frontier.png"))


def plot_figure6(e3_data: dict, output_dir: str = "outputs/results"):
    """
    Figure 6 (E8): Mechanistic Predictors of Alignment Failure.
    Scatter plots + Spearman rank correlation (rho and p-value) across all 5 diagnostics:
    1. eta_pref
    2. clash_overruled
    3. csim_surv
    4. pi_pref
    5. clash_weighted
    """
    # Collect points with diagnostics
    diag_keys = ["eta_pref", "clash_overruled", "csim_surv", "pi_pref", "clash_weighted"]
    diag_titles = [
        r"1. Effective Norm Share $\eta_{\text{pref}}$",
        r"2. Sign Overrule Rate $\text{Clash}_{\text{overruled}}$ (Top 20%)",
        r"3. Directional Survival $\text{CosSim}_{\text{surv}}$",
        r"4. Projection Recovery $\pi_{\text{pref}}$",
        r"5. Weighted Sign Clash $\text{Clash}_{\text{weighted}}$ (Fisher)"
    ]

    records = []
    for k, row in e3_data.items():
        if "diagnostics" in row and row["diagnostics"]:
            entry = {"prr": row["prr"], "method": row["method"], "N": row["N"]}
            entry.update(row["diagnostics"])
            records.append(entry)

    if not records:
        print("Warning: No diagnostic data found in E3 results. Skipping Figure 6.")
        return

    fig, axes = plt.subplots(1, 5, figsize=(25, 4.8), dpi=300)

    for idx, (d_key, title) in enumerate(zip(diag_keys, diag_titles)):
        ax = axes[idx]
        xs = [r[d_key] for r in records]
        ys = [r["prr"] for r in records]

        # Compute Spearman rank correlation
        rho, p_val = stats.spearmanr(xs, ys)

        for method in ["linear", "ties", "dare_ties"]:
            m_xs = [r[d_key] for r in records if r["method"] == method]
            m_ys = [r["prr"] for r in records if r["method"] == method]
            if m_xs:
                ax.scatter(
                    m_xs, m_ys,
                    color=COLORS[method],
                    alpha=0.75,
                    s=45,
                    label=LABELS[method] if idx == 0 else ""
                )

        # Linear trendline for visual guide
        if len(xs) > 1:
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(xs), max(xs), 50)
            ax.plot(x_line, p(x_line), color="black", linestyle="--", linewidth=1.5, alpha=0.7)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(d_key, fontsize=10)
        ax.set_ylabel("PRR (%)" if idx == 0 else "", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Correlation text annotation box
        sign_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
        ax.text(
            0.05, 0.92,
            f"Spearman $\\rho = {rho:.3f}$\n$p$ {sign_str}",
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85)
        )

    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=True, fontsize=8.5, loc="lower right")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure6_predictors_of_failure.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"📊 Saved Figure 6 (E8) to: {fig_path}")

    if os.path.exists(ARTIFACT_DIR):
        shutil.copy(fig_path, os.path.join(ARTIFACT_DIR, "figure6_predictors_of_failure.png"))


def export_table3(unified_data: dict, output_dir: str = "outputs/results"):
    """
    Compiles Table 3: Consortium Merging Master Table across N in {2, 3, 4}.
    Compares:
    - Uniform Allocation (1/N)
    - Priority-Anchored Allocation (alpha=0.40)
    - Pareto-Optimal Allocation (alpha*)
    Exports both Markdown and LaTeX formats.
    """
    methods = ["linear", "ties", "dare_ties"]
    uniform_alphas = {2: 0.50, 3: 0.3333, 4: 0.2500}

    rows = []
    for N in [2, 3, 4]:
        for method in methods:
            # 1. Uniform
            target_u = uniform_alphas[N]
            row_u = min(
                [r for r in unified_data.values() if r["N"] == N and r["method"] == method],
                key=lambda x: abs(x["alpha_pref"] - target_u),
                default=None
            )

            # 2. Priority Anchored (alpha = 0.40)
            row_p = min(
                [r for r in unified_data.values() if r["N"] == N and r["method"] == method],
                key=lambda x: abs(x["alpha_pref"] - 0.40),
                default=None
            )

            # 3. Pareto Optimal (max PRR + Mean CR sum)
            candidates = [r for r in unified_data.values() if r["N"] == N and r["method"] == method]
            row_opt = max(candidates, key=lambda x: (x["prr"] + x["mean_cr"]), default=None)

            rows.append({
                "N": N,
                "method": method,
                "u_alpha": row_u["alpha_pref"] if row_u else 0.0,
                "u_prr": row_u["prr"] if row_u else 0.0,
                "u_cr": row_u["mean_cr"] if row_u else 0.0,
                "p_prr": row_p["prr"] if row_p else 0.0,
                "p_cr": row_p["mean_cr"] if row_p else 0.0,
                "opt_alpha": row_opt["alpha_pref"] if row_opt else 0.0,
                "opt_prr": row_opt["prr"] if row_opt else 0.0,
                "opt_cr": row_opt["mean_cr"] if row_opt else 0.0
            })

    # Generate Markdown Table
    md_lines = [
        "# Table 3: Consortium Scaling Master Table ($N \\in \\{2, 3, 4\\}$)",
        "",
        "| $N$ | Merge Method | Uniform $\\alpha$ | Uniform PRR (%) | Uniform $\\overline{\\text{CR}}$ (%) | Anchored ($\\alpha=0.4$) PRR | Anchored $\\overline{\\text{CR}}$ | Pareto $\\alpha^*$ | Pareto PRR (%) | Pareto $\\overline{\\text{CR}}$ (%) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in rows:
        md_lines.append(
            f"| {r['N']} | {LABELS[r['method']]} | {r['u_alpha']:.2f} | {r['u_prr']:.1f}% | {r['u_cr']:.1f}% | {r['p_prr']:.1f}% | {r['p_cr']:.1f}% | {r['opt_alpha']:.2f} | {r['opt_prr']:.1f}% | {r['opt_cr']:.1f}% |"
        )

    md_path = os.path.join(output_dir, "table3_consortium_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"📄 Saved Table 3 (Markdown) to: {md_path}")

    # Generate LaTeX Table
    tex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Consortium Scaling Master Table ($N \\in \\{2, 3, 4\\}$) comparing Uniform ($1/N$), Priority-Anchored ($\\alpha=0.40$), and Pareto-Optimal ($\\alpha^*$) merge regimes.}",
        "\\label{tab:consortium_scaling}",
        "\\begin{tabular}{llcccccccc}",
        "\\toprule",
        "& & \\multicolumn{3}{c}{\\textbf{Uniform Allocation ($w_k=1/N$)}} & \\multicolumn{2}{c}{\\textbf{Priority-Anchored ($\\alpha=0.4$)}} & \\multicolumn{3}{c}{\\textbf{Pareto-Optimal}} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-10}",
        "$N$ & Method & $\\alpha_{\\text{pref}}$ & PRR (\\%) & $\\overline{\\text{CR}}$ (\\%) & PRR (\\%) & $\\overline{\\text{CR}}$ (\\%) & $\\alpha^*$ & PRR (\\%) & $\\overline{\\text{CR}}$ (\\%) \\\\",
        "\\midrule"
    ]
    for r in rows:
        tex_lines.append(
            f"{r['N']} & {LABELS[r['method']]} & {r['u_alpha']:.2f} & {r['u_prr']:.1f} & {r['u_cr']:.1f} & {r['p_prr']:.1f} & {r['p_cr']:.1f} & {r['opt_alpha']:.2f} & {r['opt_prr']:.1f} & {r['opt_cr']:.1f} \\\\"
        )
    tex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ]
    tex_path = os.path.join(output_dir, "table3_consortium_summary.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"📄 Saved Table 3 (LaTeX) to: {tex_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot E3 Consortium Scaling and E8 Failure Predictors")
    parser.add_argument("--e3_path", type=str, default="outputs/results/e3_sweep_results.json",
                        help="Path to E3 sweep results JSON file")
    parser.add_argument("--e2_path", type=str, default="outputs/results/e2_sweep_results.json",
                        help="Path to E2 sweep results JSON file")
    parser.add_argument("--output_dir", type=str, default="outputs/results",
                        help="Directory to save generated plots and tables")
    args = parser.parse_args()

    e2_path = args.e2_path
    e3_path = args.e3_path
    output_dir = args.output_dir

    print("\n==================================================================")
    print("🚀 Generating E3 Consortium Scaling & E8 Diagnostic Artifacts")
    print(f"   E2 Source: {e2_path}")
    print(f"   E3 Source: {e3_path}")
    print(f"   Output Directory: {output_dir}")
    print("==================================================================\n")

    e2_data = load_e2_data(e2_path)
    e3_data = {}
    if os.path.exists(e3_path):
        with open(e3_path, "r") as f:
            e3_data = json.load(f)
        print(f"Loaded {len(e3_data)} checkpoints from {e3_path}.")
    else:
        print(f"Warning: E3 sweep results not found at {e3_path}.")

    unified_data = {}
    unified_data.update(e2_data)
    unified_data.update(e3_data)
    print(f"Total unified points for analysis: {len(unified_data)}")

    if not unified_data:
        print("Error: No data available to plot.")
        return

    plot_figure3(unified_data, output_dir=output_dir)
    plot_figure5(unified_data, output_dir=output_dir)
    if e3_data:
        plot_figure6(e3_data, output_dir=output_dir)
    export_table3(unified_data, output_dir=output_dir)

    print("\n🎉 All figures and Table 3 successfully generated and synced to artifacts!\n")


if __name__ == "__main__":
    main()
