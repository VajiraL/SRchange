"""
Figures for OOD Climate-Distance Stratified Evaluation (Fig. S2a-c)
=====================================================================
Reads ood_distance_results.csv (output of ood_distance_test.py) and
produces three publication-ready figures:

  S2a — Climate shift distance (Mahalanobis) by period
  S2b — R2 by distance stratum and configuration (Poisson + absoluteerror)
  S2c — Per-period crossover: dist_mean vs delta-R2 (clustered - global), Poisson only

Usage:
    python make_ood_figures.py /path/to/ood_distance_results.csv /output/dir
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------
# Style
# ---------------------------
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
})

PERIOD_ORDER = [
    "1901_1930", "1931_1960", "1961_1990",
    "2041_2070_ssp126", "2041_2070_ssp245", "2041_2070_ssp370", "2041_2070_ssp460",
    "2071_2099_ssp126", "2071_2099_ssp245", "2071_2099_ssp370", "2071_2099_ssp460",
]
PERIOD_LABELS = {
    "1901_1930": "1901–1930", "1931_1960": "1931–1960", "1961_1990": "1961–1990",
    "2041_2070_ssp126": "2041–70\nssp126", "2041_2070_ssp245": "2041–70\nssp245",
    "2041_2070_ssp370": "2041–70\nssp370", "2041_2070_ssp460": "2041–70\nssp460",
    "2071_2099_ssp126": "2071–99\nssp126", "2071_2099_ssp245": "2071–99\nssp245",
    "2071_2099_ssp370": "2071–99\nssp370", "2071_2099_ssp460": "2071–99\nssp460",
}
PERIOD_GROUP = {
    "1901_1930": "Historical", "1931_1960": "Historical", "1961_1990": "Historical",
    "2041_2070_ssp126": "Mid-century", "2041_2070_ssp245": "Mid-century",
    "2041_2070_ssp370": "Mid-century", "2041_2070_ssp460": "Mid-century",
    "2071_2099_ssp126": "End-century", "2071_2099_ssp245": "End-century",
    "2071_2099_ssp370": "End-century", "2071_2099_ssp460": "End-century",
}
GROUP_COLORS = {
    "Historical":   "#4D7EA8",
    "Mid-century":  "#E0A458",
    "End-century":  "#C1443C",
}

STRATA_ORDER = ["Q1", "Q2", "Q3", "Q4", "ALL"]

OBJ_LABELS = {"poisson": "Poisson", "absoluteerror": "Absolute error"}
CONFIG_LABELS = {"global": "Global", "clustered": "Clustered"}

# Colors for config x objective combinations in S2b
BAR_COLORS = {
    ("poisson", "global"):        "#1D9E75",
    ("poisson", "clustered"):     "#0B6B4F",
    ("absoluteerror", "global"):  "#85B7EB",
    ("absoluteerror", "clustered"): "#378ADD",
}


def load_results(csv_path):
    df = pd.read_csv(csv_path)
    df["period"] = df["period"].astype(str)
    return df


# ---------------------------
# Fig S2a — distance by period
# ---------------------------
def fig_s2a(df, outdir):
    # Use the 'ALL' stratum rows to get per-period dist_mean reliably,
    # plus reconstruct Q1/Q3 (approx 25th/75th pct) from stratum dist ranges if available.
    sub = df[df["stratum"] == "ALL"][["period", "dist_mean"]].drop_duplicates()
    sub = sub.set_index("period").reindex(PERIOD_ORDER).reset_index()

    # Try to get a spread estimate from per-stratum dist_lo/dist_hi (Q1 lo -> Q4 hi covers full range)
    spread = (df[df["stratum"].isin(["Q1", "Q4"])]
              .groupby("period")
              .agg(lo=("dist_lo", "min"), hi=("dist_hi", "max"))
              .reindex(PERIOD_ORDER))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    y_pos = np.arange(len(PERIOD_ORDER))[::-1]   # historical at top

    for i, period in enumerate(PERIOD_ORDER):
        y = y_pos[i]
        color = GROUP_COLORS[PERIOD_GROUP[period]]
        dmean = sub.loc[sub["period"] == period, "dist_mean"].values
        dmean = dmean[0] if len(dmean) else np.nan

        if period in spread.index and not spread.loc[period].isna().any():
            lo, hi = spread.loc[period, "lo"], spread.loc[period, "hi"]
            ax.plot([lo, hi], [y, y], color=color, lw=2.5, alpha=0.35, solid_capstyle="round")

        ax.scatter([dmean], [y], color=color, s=70, zorder=3, edgecolor="white", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([PERIOD_LABELS[p].replace("\n", " ") for p in PERIOD_ORDER], fontsize=9)
    ax.set_xlabel("Mahalanobis climate shift distance from 1991–2020")
    ax.set_title("Climate shift distance by target period", fontsize=11, loc="left")
    ax.axvline(0, color="#999999", lw=0.8, ls=":")

    # Legend for groups
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=g)
               for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    out = os.path.join(outdir, "FigS2a_distance_by_period.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------
# Fig S2b — R2 by stratum and config
# ---------------------------
def fig_s2b(df, outdir):
    sub = df[df["stratum"].isin(STRATA_ORDER)].copy()
    agg = (sub.groupby(["objective", "config", "stratum"])["R2"]
           .mean()
           .reset_index())

    fig, ax = plt.subplots(figsize=(7.5, 5))

    n_strata = len(STRATA_ORDER)
    combos = [("poisson", "global"), ("poisson", "clustered"),
              ("absoluteerror", "global"), ("absoluteerror", "clustered")]
    n_combo = len(combos)
    width = 0.8 / n_combo
    x = np.arange(n_strata)

    for ci, (obj, cfg) in enumerate(combos):
        vals = []
        for s in STRATA_ORDER:
            row = agg[(agg["objective"] == obj) & (agg["config"] == cfg) & (agg["stratum"] == s)]
            vals.append(row["R2"].values[0] if len(row) else np.nan)
        offset = (ci - (n_combo - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92,
                      color=BAR_COLORS[(obj, cfg)],
                      label=f"{OBJ_LABELS[obj]} – {CONFIG_LABELS[cfg]}")

    ax.set_xticks(x)
    ax.set_xticklabels(STRATA_ORDER)
    ax.set_ylabel("R² (mean across periods)")
    ax.set_xlabel("Climate shift distance stratum")
    ax.set_title("Prediction skill by climate shift distance and configuration",
                 fontsize=11, loc="left")
    ax.set_ylim(0, max(0.8, agg["R2"].max() * 1.1))
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower center",
             bbox_to_anchor=(0.5, -0.32))

    fig.tight_layout()
    out = os.path.join(outdir, "FigS2b_R2_by_stratum_config.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------
# Fig S2c — crossover scatter
# ---------------------------
def fig_s2c(df, outdir):
    sub = df[(df["stratum"] == "ALL") & (df["objective"] == "poisson")].copy()
    piv = sub.pivot_table(index="period", columns="config", values=["R2", "dist_mean"])

    periods = [p for p in PERIOD_ORDER if p in piv.index]
    dist_mean = piv["dist_mean"]["global"].reindex(periods)  # same dist regardless of config
    delta_r2  = (piv["R2"]["clustered"] - piv["R2"]["global"]).reindex(periods)

    fig, ax = plt.subplots(figsize=(7, 5.2))

    for period in periods:
        grp = PERIOD_GROUP[period]
        color = GROUP_COLORS[grp]
        ax.scatter(dist_mean[period], delta_r2[period], s=110, color=color,
                  edgecolor="white", linewidth=0.8, zorder=3)
        label = PERIOD_LABELS[period].replace("\n", " ")
        ax.annotate(label, (dist_mean[period], delta_r2[period]),
                   textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                   color="#333333")

    ax.axhline(0, color="#999999", lw=1.0, ls="--", zorder=1)
    ax.set_xlabel("Mahalanobis climate shift distance (mean)")
    ax.set_ylabel("ΔR² (clustered − global), Poisson")
    ax.set_title("Crossover point: clustered vs global advantage by climate distance",
                 fontsize=11, loc="left")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=g)
               for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    # Shaded regions for interpretation
    ax.axhspan(0, ax.get_ylim()[1], color=GROUP_COLORS["Historical"], alpha=0.04, zorder=0)
    ax.text(0.02, 0.96, "clustered better", transform=ax.transAxes,
           fontsize=8, color="#555555", va="top")
    ax.text(0.02, 0.04, "global better", transform=ax.transAxes,
           fontsize=8, color="#555555", va="bottom")

    fig.tight_layout()
    out = os.path.join(outdir, "FigS2c_crossover_distance_vs_deltaR2.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------
# Main
# ---------------------------
def main():
    #if len(sys.argv) < 2:
    #    print("Usage: python make_ood_figures.py /path/to/ood_distance_results.csv [output_dir]")
    #    sys.exit(1)

    csv_path = '/home/vajira/SRchange/output_SR/ood_distance_test/ood_distance_results.csv'# sys.argv[1]
    outdir   = '/home/vajira/SRchange/plots'# sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path) or "."
    os.makedirs(outdir, exist_ok=True)

    df = load_results(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")

    fig_s2a(df, outdir)
    fig_s2b(df, outdir)
    fig_s2c(df, outdir)

    print("All figures generated.")


if __name__ == "__main__":
    main()