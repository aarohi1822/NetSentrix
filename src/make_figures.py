"""Build the README/LinkedIn figures from the saved evaluation files.

    python src/make_figures.py

Every plotted number is read from data/processed/evaluation/{smap,msl}_benchmark_summary.json
and cross-checked against results/metrics.json. If the two saved sources disagree on any
plotted value, the script exits with an error and writes nothing.

Outputs: assets/tradeoff.png, assets/attribution.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATASETS = ("SMAP", "MSL")
DETECTORS = (
    # key in the result files, label, categorical slot colour, marker (secondary encoding)
    ("lstm_autoencoder", "LSTM-AE", "#2a78d6", "o"),
    ("isolation_forest", "Isolation Forest", "#eb6834", "s"),
    ("first_difference", "|x$_t$ − x$_{t−1}$|", "#1baf7a", "D"),
)
ANOMALY_TYPES = (("spike", "Spike"), ("level_shift", "Level shift"), ("drift", "Gradual drift"))

# Reference palette, light mode (dataviz skill references/palette.md).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BAR = "#2a78d6"


class SourceMismatch(SystemExit):
    pass


def load_values(eval_dir: Path, metrics_path: Path) -> dict:
    """Read every value the figures need, from both saved sources, and require them to agree."""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["datasets"]
    values: dict = {}
    mismatches: list[str] = []

    def take(label: str, primary, secondary):
        if primary != secondary:
            mismatches.append(f"{label}: summary={primary!r} metrics.json={secondary!r}")
        return primary

    for ds in DATASETS:
        summary = json.loads((eval_dir / f"{ds.lower()}_benchmark_summary.json").read_text(encoding="utf-8"))
        mj = metrics[ds]
        entry = {"channels": take(f"{ds}.channel_count", summary["channel_count"], mj["channel_count"])}
        entry["detectors"] = {}
        for key, *_ in DETECTORS:
            pooled = mj["detectors"][key]["pooled"]
            entry["detectors"][key] = {
                field: take(f"{ds}.{key}.{field}", summary[key][field], pooled[field])
                for field in ("detected_events", "events", "false_alarms_per_1k")
            }
        entry["attribution"] = {}
        for kind, _ in ANOMALY_TYPES:
            block, mblock = summary["attribution"]["by_type"][kind], mj["attribution"]["by_type"][kind]
            entry["attribution"][kind] = {
                field: take(f"{ds}.attribution.{kind}.{field}", block[field], mblock[field])
                for field in ("top3_hit_rate", "top3_chance", "n")
            }
        values[ds] = entry

    if mismatches:
        raise SourceMismatch("Saved result files disagree; no figures written:\n  " + "\n  ".join(mismatches))
    return values


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0, pad=6)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(title: str, subtitle: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200, facecolor=SURFACE)
    fig.text(0.04, 0.955, title, fontsize=14, fontweight="bold", color=TEXT_PRIMARY, ha="left", va="top")
    fig.text(0.04, 0.895, subtitle, fontsize=9, color=TEXT_SECONDARY, ha="left", va="top")
    return fig, axes


def plot_tradeoff(values: dict, out: Path) -> None:
    fig, axes = new_figure(
        "Catching events vs. raising false alarms",
        "NASA SMAP & MSL test sets · pooled over all channels · one run (seed 42) · "
        "same threshold rule (99th pct of training scores) for all three detectors",
    )
    y_max = max(d["false_alarms_per_1k"] for v in values.values() for d in v["detectors"].values())
    for ax, ds in zip(axes, DATASETS):
        style_axes(ax)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        events = values[ds]["detectors"][DETECTORS[0][0]]["events"]
        points = {key: values[ds]["detectors"][key] for key, *_ in DETECTORS}
        x_span, y_span = events * 1.38, y_max * 1.18
        for key, label, colour, marker in DETECTORS:
            d = points[key]
            x, y = d["detected_events"], d["false_alarms_per_1k"]
            ax.scatter(x, y, s=110, color=colour, marker=marker, edgecolors=SURFACE, linewidths=2,
                       zorder=3, label=label)
            # Label to the right unless another point sits in that space; then label to the left.
            blocked = any(
                0 < o["detected_events"] - x < 0.35 * x_span and abs(o["false_alarms_per_1k"] - y) < 0.12 * y_span
                for k, o in points.items() if k != key
            )
            ax.annotate(
                f"{label}\n{x}/{events} events · {y:.2f}",
                (x, y), xytext=(-10, 0) if blocked else (10, 0), textcoords="offset points",
                fontsize=8.5, color=TEXT_PRIMARY, va="center", ha="right" if blocked else "left",
                bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"}, zorder=4,
            )
        ax.axvline(events, color=TEXT_SECONDARY, linewidth=0.9, linestyle=(0, (2, 3)), zorder=2)
        ax.text(events, y_span * 0.985, f" all {events}", fontsize=8, color=TEXT_SECONDARY, ha="left", va="top")
        ax.set_xlim(0, x_span)
        ax.set_ylim(0, y_span)
        ax.set_title(f"{ds} · {values[ds]['channels']} channels · {events} labelled events",
                     fontsize=10.5, color=TEXT_PRIMARY, loc="left", pad=10)
        ax.set_xlabel("Anomaly events detected (higher is better)", fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylabel("False alarms per 1,000 points (lower is better)", fontsize=9, color=TEXT_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=TEXT_PRIMARY, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.04, 0.02, "Source: data/processed/evaluation/*_benchmark_summary.json",
             fontsize=7, color=TEXT_SECONDARY, ha="left")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.19, wspace=0.22)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_attribution(values: dict, out: Path) -> None:
    fig, axes = new_figure(
        "Root-cause attribution: is the injected input in the top 3?",
        "1,000 synthetic anomalies injected per dataset into random inputs and timesteps · "
        "hit = injected input ranks top-3 by LSTM-AE reconstruction error",
    )
    for ax, ds in zip(axes, DATASETS):
        style_axes(ax)
        attr = values[ds]["attribution"]
        rates = [attr[k]["top3_hit_rate"] for k, _ in ANOMALY_TYPES]
        labels = [f"{name}\n(n={attr[k]['n']})" for k, name in ANOMALY_TYPES]
        chances = {attr[k]["top3_chance"] for k, _ in ANOMALY_TYPES}
        chance = max(chances)
        if max(chances) - min(chances) > 1e-9:
            raise SystemExit(f"{ds}: chance level differs across anomaly types: {sorted(chances)}")
        positions = range(len(rates))
        bars = ax.bar(positions, rates, width=0.5, color=BAR, edgecolor=SURFACE, linewidth=2, zorder=3)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.025, f"{rate:.2f}",
                    ha="center", va="bottom", fontsize=9.5, color=TEXT_PRIMARY)
        ax.axhline(chance, color=TEXT_SECONDARY, linestyle=(0, (5, 4)), linewidth=1.4, zorder=4)
        ax.text(1.015, chance, f"random\nchance\n{chance:.2f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color=TEXT_SECONDARY, clip_on=False)
        ax.set_xticks(list(positions), labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{ds} · {values[ds]['channels']} channels", fontsize=10.5, color=TEXT_PRIMARY,
                     loc="left", pad=10)
        ax.set_ylabel("Top-3 hit rate", fontsize=9, color=TEXT_SECONDARY)
    fig.text(0.96, 0.02, "Source: data/processed/evaluation/*_benchmark_summary.json",
             fontsize=7, color=TEXT_SECONDARY, ha="right")
    fig.subplots_adjust(left=0.07, right=0.93, top=0.78, bottom=0.17, wspace=0.28)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-dir", default="data/processed/evaluation")
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--out-dir", default="assets")
    args = parser.parse_args()

    values = load_values(Path(args.eval_dir), Path(args.metrics))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_tradeoff(values, out_dir / "tradeoff.png")
    plot_attribution(values, out_dir / "attribution.png")
    print(f"Wrote {out_dir / 'tradeoff.png'} and {out_dir / 'attribution.png'} "
          "(all values matched between the summary files and results/metrics.json)")


if __name__ == "__main__":
    main()
