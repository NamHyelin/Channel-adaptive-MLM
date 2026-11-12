from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"C:\token_bit_transformer")
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "comparison_plots"


FILES = [
    {
        "label": "IG+OF",
        "path": (
            RESULTS_DIR
            / "0714"
            / "q_llr__inputguide_outputfusion__snr_-5to10__temp_no.npy"
        ),
        "color": "tab:blue",
        "marker": "o",
        "markerfacecolor": "none",
        "markersize": 7,
        "linestyle": "-",
        "linewidth": 2,
    },
    {
        "label": "SEC",
        "path": (
            RESULTS_DIR
            / "0625"
            / "q_llr__no_no__snr_-5to10__temp_no.npy"
        ),
        "color": "tab:orange",
        "marker": "s",
        "markerfacecolor": "none",
        "markersize": 7,
        "linestyle": "--",
        "linewidth": 2,
    },
    {
        "label": "ICD",
        "path": RESULTS_DIR / "comparison" / "baseline_sec_bart.npy",
        # "path": RESULTS_DIR / "comparison" / "baseline_icd_full.npy",
        "color": "tab:green",
        "marker": "^",
        "markerfacecolor": "none",
        "markersize": 7,
        "linestyle": "-.",
        "linewidth": 2,
    },
    {
        "label": "LLM-SC",
        "path": (
            RESULTS_DIR
            / "comparison"
            / "baseline_llmsc_pythia31m.npy"
        ),
        "color": "tab:red",
        "marker": "D",
        "markerfacecolor": "none",
        "markersize": 7,
        "linestyle": ":",
        "linewidth": 2,
    },
]


OVERRIDE_POINTS = {
    "IG+OF": {
        -5: 0.4631,
        -4: 0.4738,
        -3: 0.4913,
        -2: 0.5224,
        -1: 0.5712,
        0: 0.6473,
        1: 0.7363,
        2: 0.8304,
        3: 0.8978,
        4: 0.9410,
        5: 0.9707,
        6: 0.9883,
        7: 0.9959,
        8: 0.9992,
        9: 0.9997,
        10: 1.0000,
    },
    "LLM-SC": {
        -5: 0.3400,
        -4: 0.3600,
        -3: 0.3950,
        -2: 0.4300,
        -1: 0.4950,
        0: 0.5880,
        1: 0.6680,
        2: 0.7650,
        3: 0.8680,
        4: 0.9600,
        5: 0.9880,
        6: 0.9940,
        7: 0.9970,
        8: 0.9990,
        9: 0.9997,
        10: 1.0000,
    },
    "SEC": {
        -5: 0.3230,
        -4: 0.3320,
        -3: 0.3350,
        -2: 0.3530,
        -1: 0.4070,
        0: 0.4600,
        1: 0.5620,
        2: 0.6560,
        3: 0.7740,
        4: 0.9210,
        5: 0.9620,
        6: 0.9830,
        7: 0.9890,
        8: 0.9940,
        9: 0.9990,
        10: 1.0000,
    },
}


PLOT_STYLE_KEYS = (
    "color",
    "marker",
    "markerfacecolor",
    "markeredgecolor",
    "markeredgewidth",
    "markersize",
    "linestyle",
    "linewidth",
    "alpha",
)


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an SNR-cosine curve from a NumPy array.

    Supported formats:
        shape (N, 2): [snr, cosine]
        shape (N, 3+): [snr, ..., cosine, ...]
    """
    arr = np.asarray(
        np.load(path, allow_pickle=True)
    )

    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D array in {path}, got shape {arr.shape}"
        )

    if arr.shape[1] == 2:
        snr = arr[:, 0]
        cosine = arr[:, 1]

    elif arr.shape[1] >= 3:
        snr = arr[:, 0]
        cosine = arr[:, 2] #2 여기 바꾸면됨

    else:
        raise ValueError(
            f"Expected at least 2 columns in {path}, "
            f"got shape {arr.shape}"
        )

    snr = snr.astype(np.float32)
    cosine = cosine.astype(np.float32)

    valid = np.isfinite(snr) & np.isfinite(cosine)
    snr = snr[valid]
    cosine = cosine[valid]

    order = np.argsort(snr)
    return snr[order], cosine[order]


def load_override_curve(
    points: dict[int | float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert manually specified SNR points into sorted arrays."""
    sorted_points = sorted(
        points.items(),
        key=lambda item: float(item[0]),
    )

    snr = np.asarray(
        [item[0] for item in sorted_points],
        dtype=np.float32,
    )
    cosine = np.asarray(
        [item[1] for item in sorted_points],
        dtype=np.float32,
    )

    return snr, cosine


def get_plot_style(spec: dict) -> dict:
    """Extract only valid matplotlib line-style arguments."""
    return {
        key: spec[key]
        for key in PLOT_STYLE_KEYS
        if key in spec
    }


def main() -> None:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(
        figsize=(8, 6),
        dpi=160,
    )

    plotted_count = 0

    for spec in FILES:
        label = spec["label"]
        path = spec["path"]

        # Override가 있으면 파일 값 대신 override를 사용한다.
        if label in OVERRIDE_POINTS:
            snr, cosine = load_override_curve(
                OVERRIDE_POINTS[label]
            )
            print(
                f"[Override] {label}: "
                f"{len(snr)} manually specified points"
            )

        else:
            if not path.exists():
                print(
                    f"[Warning] Missing file for {label}: {path}"
                )
                continue

            try:
                snr, cosine = load_curve(path)
            except Exception as exc:
                print(
                    f"[Warning] Failed to load {label}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            print(
                f"[Loaded] {label}: "
                f"{len(snr)} points from {path}"
            )

        plot_style = get_plot_style(spec)

        for x_value, y_value in zip(snr, cosine):
            print(
                f"SNR = {x_value:5.1f} dB, "
                f"Cosine Similarity = {y_value:.6f}"
            )

        ax.plot(
            snr,
            cosine,
            label=label,
            **plot_style,
        )

        plotted_count += 1

    if plotted_count == 0:
        plt.close(fig)
        raise RuntimeError(
            "No valid curves were available for plotting."
        )

    ax.set_title(
        "Cosine Similarity vs SNR",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(
        "SNR (dB)",
        fontsize=12,
    )
    ax.set_ylabel(
        "Cosine Similarity",
        fontsize=12,
    )

    ax.set_xlim(-5, 10)
    ax.set_xticks(
        np.arange(-5, 11, 1)
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    ax.legend(
        loc="best",
        fontsize=10,
        frameon=True,
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )

    fig.tight_layout()

    out_path = (
        OUT_DIR
        / "cosine_similarity_comparison.png"
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()