"""Distribution plots of RHMI/EHMI across units for the two example sessions
(001_compute_example_units.py output): overall, and broken down by
`area_group` (coarse area parcellation, reused from
reports/ssl_analysis/derived/unit_area_labels.parquet -- see
ssl_analysis_patterns.md's Area Grouping section). Areas with fewer than
MIN_UNITS_PER_AREA valid (non-NaN) index values are dropped from the
per-area panel to avoid single-unit "distributions".
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("projects/ssl-history-modulation-indices/exploratory-analyses")
MIN_UNITS_PER_AREA = 5


def main():
    df = pd.read_parquet(OUT_DIR / "example_unit_indices.parquet")
    areas = pd.read_parquet("reports/ssl_analysis/derived/unit_area_labels.parquet")[
        ["session_id", "cluster_id", "area_group", "area_acronym_custom"]
    ]
    df = df.merge(areas, on=["session_id", "cluster_id"], how="left")
    print(f"{len(df)} units, {df['area_group'].isna().sum()} missing area_group after merge")

    session_labels = {
        "MH030_20250501_151231": "MH030_20250501_151231 (R+)",
        "MH023_20250316_110814": "MH023_20250316_110814 (R-)",
    }
    sessions = list(session_labels)

    # --- Overall: RHMI/EHMI distribution per session ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, metric in zip(axes, ("rhmi_index", "ehmi_index")):
        data = [df.loc[df.session_id == sid, metric].dropna() for sid in sessions]
        bp = ax.boxplot(data, tick_labels=[session_labels[s] for s in sessions], showfliers=True, widths=0.5)
        for i, d in enumerate(data, start=1):
            jitter = 0.08 * (pd.Series(range(len(d))) % 2 * 2 - 1) * 0.5
            ax.scatter([i] * len(d) + jitter.to_numpy() * 0, d, alpha=0.25, s=8, color="tab:blue")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{metric.split('_')[0].upper()} (n={[len(d) for d in data]})")
        ax.set_xticklabels([session_labels[s] for s in sessions], rotation=15, ha="right")
    axes[0].set_ylabel("Index")
    fig.suptitle("RHMI / EHMI distribution across units, overall")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = OUT_DIR / "003_index_distributions_overall.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # --- Per area, one figure per session ---
    for sid in sessions:
        sub = df[df.session_id == sid]
        area_counts = sub.groupby("area_group")[["rhmi_index", "ehmi_index"]].apply(lambda g: g.notna().sum())
        keep_areas = area_counts.index[(area_counts["rhmi_index"] >= MIN_UNITS_PER_AREA) | (area_counts["ehmi_index"] >= MIN_UNITS_PER_AREA)]
        keep_areas = [a for a in keep_areas if pd.notna(a)]
        # order areas by median RHMI for readability
        order = sub[sub.area_group.isin(keep_areas)].groupby("area_group")["rhmi_index"].median().sort_values().index.tolist()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        for ax, metric in zip(axes, ("rhmi_index", "ehmi_index")):
            data = [sub.loc[sub.area_group == a, metric].dropna() for a in order]
            ns = [len(d) for d in data]
            ax.boxplot(data, tick_labels=[f"{a} (n={n})" for a, n in zip(order, ns)], orientation="horizontal", showfliers=True, widths=0.6)
            for i, d in enumerate(data, start=1):
                ax.scatter(d, [i] * len(d), alpha=0.3, s=8, color="tab:blue")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_title(metric.split("_")[0].upper())
            ax.set_xlabel("Index")
        fig.suptitle(f"{session_labels[sid]} -- index by area_group (min {MIN_UNITS_PER_AREA} units/area)")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out_path = OUT_DIR / f"003_index_by_area_{sid}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
