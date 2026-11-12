"""
Plot SNR-versus-cosine-similarity curves from the user's method and the three
aligned paper baselines.

Expected default `.npy` layout from all four evaluation scripts:
    column 0: SNR (dB)
    column 1: mean F1
    column 2: mean cosine similarity
    column 3: mean BERTScore

Recommended usage:
    python plot_cosine_comparison.py \
        --ours results/ours.npy \
        --sec results/baseline_sec_bart.npy \
        --icd results/baseline_icd.npy \
        --llmsc results/baseline_llmsc.npy \
        --output results/cosine_vs_snr.png

You may also add arbitrary curves:
    --series "Ablation A=path/to/a.npy" --series "Ablation B=path/to/b.npy"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare cosine similarity over SNR")
    p.add_argument("--ours", help="Numpy result file for the proposed method")
    p.add_argument("--sec", help="Numpy result file from baseline_sec_bart.py")
    p.add_argument("--icd", help="Numpy result file from baseline_icd.py")
    p.add_argument("--llmsc", help="Numpy result file from baseline_llmsc.py")
    p.add_argument(
        "--series",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional curve; may be repeated",
    )
    p.add_argument("--snr-column", type=int, default=0)
    p.add_argument("--cosine-column", type=int, default=2)
    p.add_argument("--output", default="cosine_similarity_vs_snr.png")
    p.add_argument("--title", default="Semantic similarity over AWGN channel")
    p.add_argument("--xlabel", default="SNR (dB)")
    p.add_argument("--ylabel", default="Cosine similarity")
    p.add_argument("--fig-width", type=float, default=8.0)
    p.add_argument("--fig-height", type=float, default=6.0)
    p.add_argument("--font-size", type=float, default=13.0)
    p.add_argument("--tick-size", type=float, default=12.0)
    p.add_argument("--line-width", type=float, default=2.0)
    p.add_argument("--marker-size", type=float, default=6.0)
    p.add_argument("--x-min", type=float)
    p.add_argument("--x-max", type=float)
    p.add_argument("--y-min", type=float, default=0.0)
    p.add_argument("--y-max", type=float, default=1.02)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def parse_series(args: argparse.Namespace) -> list[tuple[str, Path]]:
    series: list[tuple[str, Path]] = []
    fixed = [
        ("Proposed", args.ours),
        ("SEC-BART", args.sec),
        ("ICD", args.icd),
        ("LLM-SC", args.llmsc),
    ]
    for label, value in fixed:
        if value:
            series.append((label, Path(value)))

    for item in args.series:
        if "=" not in item:
            raise ValueError(f"--series must use LABEL=PATH format, got: {item!r}")
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid --series value: {item!r}")
        series.append((label, Path(path)))

    if not series:
        raise ValueError("Provide at least one of --ours/--sec/--icd/--llmsc/--series.")
    return series


def numeric_array_from_object(array: np.ndarray) -> np.ndarray:
    """Support numeric matrices and simple object arrays of row dictionaries."""
    if array.dtype != object:
        return np.asarray(array, dtype=float)

    rows = array.tolist()
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Object array is empty or unsupported.")

    if all(isinstance(row, dict) for row in rows):
        parsed = []
        for row in rows:
            snr = row.get("snr_db")
            cosine = row.get("mean_cosine_similarity", row.get("cosine_similarity"))
            if snr is None or cosine is None:
                continue
            parsed.append([float(snr), float("nan"), float(cosine), float("nan")])
        if not parsed:
            raise ValueError("Could not find snr_db and cosine fields in object array.")
        return np.asarray(parsed, dtype=float)

    return np.asarray(rows, dtype=float)


def load_curve(path: Path, snr_column: int, cosine_column: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.load(path, allow_pickle=True)
    array = numeric_array_from_object(raw)
    if array.ndim == 1:
        if array.size <= max(snr_column, cosine_column):
            raise ValueError(f"Not enough columns in {path}: shape={array.shape}")
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] <= max(snr_column, cosine_column):
        raise ValueError(
            f"Expected a 2-D result array with columns {snr_column} and "
            f"{cosine_column}; got {array.shape} from {path}"
        )

    snr = array[:, snr_column].astype(float)
    cosine = array[:, cosine_column].astype(float)
    valid = np.isfinite(snr) & np.isfinite(cosine)
    snr = snr[valid]
    cosine = cosine[valid]
    order = np.argsort(snr)
    return snr[order], cosine[order]


def main() -> None:
    args = parse_args()
    curves = parse_series(args)

    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.labelsize": args.font_size,
            "axes.titlesize": args.font_size + 1,
            "legend.fontsize": args.font_size - 1,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
        }
    )

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    markers = ("o", "s", "^", "D", "x", "v", "P", "*")
    all_snr: list[float] = []

    for index, (label, path) in enumerate(curves):
        snr, cosine = load_curve(path, args.snr_column, args.cosine_column)
        all_snr.extend(snr.tolist())
        ax.plot(
            snr,
            cosine,
            marker=markers[index % len(markers)],
            linewidth=args.line_width,
            markersize=args.marker_size,
            label=label,
        )

    x_min = args.x_min if args.x_min is not None else min(all_snr)
    x_max = args.x_max if args.x_max is not None else max(all_snr)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(args.y_min, args.y_max)

    # Integer SNR sweeps are the default in the user's experiment.
    if float(x_min).is_integer() and float(x_max).is_integer() and x_max - x_min <= 30:
        ax.set_xticks(np.arange(int(x_min), int(x_max) + 1, 1))

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved comparison plot: {output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
