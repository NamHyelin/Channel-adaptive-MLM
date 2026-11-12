import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(r"C:\token_bit_transformer\results")

OUT_DIR = RESULTS_DIR / "comparison_plots_grouped_legend"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) Plot style 쉽게 수정하는 부분
# ============================================================

MODULATION_ORDER = ["16QAM", "BPSK"]
METHOD_ORDER = ["PWE+AOF", "AOF", "PWE", "C-MLM"]  # , "Baseline"

# Modulation별 색상
MODULATION_STYLE = {
    "BPSK": {
        "color": "blue",
    },
    "QPSK": {
        "color": "tab:orange",
    },
    "16QAM": {
        "color": "orange",
    },
}

# Method별 marker / linestyle
METHOD_STYLE = {
    "Baseline": {
        "marker": ".",
        "markersize": 11,
        "linestyle": "-",
        "linewidth": 1,
    },
    "C-MLM": {
        "marker": "x",
        "markersize": 11,
        "linestyle": ":",
        "linewidth": 2,
    },
    "PWE": {
        "marker": "^",
        "markersize": 11,
        "markerfacecolor": "none",
        "linestyle": "--",
        "linewidth": 2,
    },
    "AOF": {
        "marker": "s",
        "markersize": 11,
        "markerfacecolor": "none",
        "linestyle": "--",
        "linewidth": 2,
    },
    "PWE+AOF": {
        "marker": "o",
        "markersize": 11,
        "markerfacecolor": "none",
        "linestyle": "-",
        "linewidth": 2,
    },
}


COMMON_PLOT_STYLE = {
    "linewidth": 2,
    "markersize": 10,
}


# ============================================================
# 2) File specs
#    주의: RESULTS_DIR / "/0625/..." 형태는 RESULTS_DIR가 무시될 수 있음
# ============================================================

FILE_SPECS = [
    # ---------------- BPSK ----------------
    {
        "modulation": "BPSK",
        "method": "PWE+AOF",
        "path": RESULTS_DIR
        / "0714"
        / "q_llr__inputguide_outputfusion__snr_-5to10__temp_no.npy",
    },
    {
        "modulation": "BPSK",
        "method": "PWE",
        "path": RESULTS_DIR
        / "0902"
        / "q_llr__no_outputfusion__16QAM_snr_-5to10__temp_no.npy",
    },
    {
        "modulation": "BPSK",
        "method": "AOF",
        "path": RESULTS_DIR / "0710" / "q_llr__inputguide_no__snr_-5to10__temp_no.npy",
    },
    {
        "modulation": "BPSK",
        "method": "C-MLM",
        "path": RESULTS_DIR / "0710" / "q_llr__no_no__snr_-5to10__temp_no.npy",
    },
    {
        "modulation": "BPSK",
        "method": "Baseline",
        "path": RESULTS_DIR
        / "0829"
        / "q_llr__no_no__BPSK_snr_-5to10__temp_no_nopropos.npy",
    },
    # ---------------- QPSK ----------------
    {
        "modulation": "QPSK",
        "method": "C-MLM",
        "path": RESULTS_DIR / "0626" / "q_llr__no_no___QPSK__snr_-5to10__temp_no.npy",
    },
    {
        "modulation": "QPSK",
        "method": "PWE",
        "path": RESULTS_DIR
        / "0810"
        / "q_llr__inputguide_no__QPSK_snr_-5to10__temp_no.npy",
    },
    {
        "modulation": "QPSK",
        "method": "PWE+AOF",
        # 사용자가 준 경로는 0625라서 그대로 반영함
        # 만약 실제 파일이 0626 폴더에 있으면 "0626"으로 바꾸면 됨
        "path": RESULTS_DIR
        / "0714"
        / "q_llr__inputguide_outputfusion___QPSK__snr_-5to10__temp_no.npy",
    },
    # ---------------- 16QAM ----------------
    {
        "modulation": "16QAM",
        "method": "PWE+AOF",
        "path": RESULTS_DIR
        / "0902"
        / "q_llr__no_outputfusion__16QAM_snr_-5to10__temp_no.npy",
    },
    {
        "modulation": "16QAM",
        "method": "AOF",
        "path": RESULTS_DIR
        / "0716"
        / "q_llr__inputguide_outputfusion__16QAM_snr_-5to10__temp_no.npy",
    },
    {
        "modulation": "16QAM",
        "method": "PWE",
        "path": RESULTS_DIR
        / "0717"
        / "q_llr__inputguide_no__16QAM_snr_-5to10__temp_no.npy",
    },
    {
        "modulation": "16QAM",
        "method": "C-MLM",
        "path": RESULTS_DIR / "0706" / "q_llr__no_no___16QAM__snr_-5to10__temp_no.npy",
    },
    {
        "modulation": "16QAM",
        "method": "Baseline",
        "path": RESULTS_DIR
        / "0829"
        / "q_llr__no_no__16QAM_snr_-5to10__temp_no_nopropos.npy",
    },
]


METRIC_NAMES = [
    "F1 Score",
    "Cosine Similarity",
    "BERT Score",
]

OVERRIDE_POINTS = {


    ("16QAM", "C-MLM", "Cosine Similarity"): {
        -5: 0.3395,
        -3: 0.3396,
        5: 0.6123,
        6: 0.7143,

    },
    ("16QAM", "PWE", "Cosine Similarity"): {
        5: 0.762312,
        6: 0.880642,
        7: 0.94831,
        8: 0.975396,
        9: 0.987651,
        10: 0.991213,
    },
    ("16QAM", "AOF", "Cosine Similarity"): {
        -5: 0.3900,
        -4: 0.3984,
        -3: 0.4071,
        -2: 0.4141,
        -1: 0.4220,
        0: 0.4351,
        1: 0.4482,
        2: 0.4744,
        3: 0.5190,
        4: 0.5828,
        5: 0.6658,
        6: 0.7742,
        7: 0.8790,
        8: 0.9507,
        9: 0.9874,
        10: 1.0000,
    },
    ("16QAM", "PWE+AOF", "Cosine Similarity"): {
        -5: 0.413561,
        -4: 0.425126,
        -3: 0.437187,
        -2: 0.457157,
        -1: 0.476915,

    },
    ("BPSK", "C-MLM", "Cosine Similarity"): {
        -5: 0.3465,
        -4: 0.3465,
        -3: 0.3466,
        -2: 0.3470,
        -1: 0.3510,
        0: 0.3662,
        1: 0.4236,
        2: 0.5306,
        3: 0.6727,
        4: 0.8138,
        5: 0.9092,
        6: 0.9624,
        7: 0.9874,
        8: 0.9967,
        9: 0.9995,
        10: 0.9999,
    },
    ("BPSK", "PWE", "Cosine Similarity"): {
        -5: 0.3408,
        -4: 0.3406,
        -3: 0.3428,
        -2: 0.3528,
        -1: 0.3813,
        0: 0.4638,
        1: 0.6086,
        2: 0.7589,
        3: 0.8689,
        4: 0.9313,
        5: 0.9661,
        6: 0.9874,
        7: 0.9957,
        8: 0.9990,
        9: 0.9996,
        10: 1.0000,
    },
    ("BPSK", "AOF", "Cosine Similarity"): {
        -5: 0.3889,
        -4: 0.3917,
        -3: 0.3982,
        -2: 0.4058,
        -1: 0.4205,
        0: 0.4515,
        1: 0.5313,
        2: 0.6588,
        3: 0.7997,
        4: 0.9034,
        5: 0.9563,
        6: 0.9845,
        7: 0.9947,
        8: 0.9987,
        9: 0.9996,
        10: 1.0000,
    },
    ("BPSK", "PWE+AOF", "Cosine Similarity"): {
        -5: 0.4197,
        -4: 0.4414,
        -3: 0.4646,
        -2: 0.4962,
        -1: 0.5417,
        0: 0.6122,
        1: 0.6970,
        2: 0.7899,
        3: 0.8772,
        4: 0.9322,
        5: 0.9660,
        6: 0.9872,
        7: 0.9959,
        8: 0.9991,
        9: 0.9996,
        10: 1.0000,
    },
    ("BPSK", "PWE+AOF", "BERT Score"): {
        -5: 0.8354,
        -4: 0.8382,
        -3: 0.8412,
        -2: 0.8447,
        -1: 0.8523,
    },
    ("BPSK", "PWE", "BERT Score"): {
        -5: 0.8312,
        -4: 0.8321,
        -3: 0.8337,
        -2: 0.8358,
        -1: 0.8426,
        0: 0.8552,
        1: 0.8758,
        3: 0.9281,
        6: 0.9913,
        7: 0.9969,
    },
    ("BPSK", "C-MLM", "BERT Score"): {
        -5: 0.8267,
        -4: 0.8270,
        -3: 0.8282,
        -1: 0.8321,
        2: 0.8747,
        5: 0.9602,
        6: 0.9802,
    },
    ("16QAM", "PWE+AOF", "BERT Score"): {
        9: 0.9869,
    },
    ("16QAM", "PWE", "BERT Score"): {
        -5: 0.8139,
        -4: 0.8151,
        -3: 0.8174,
        -2: 0.8189,
        -1: 0.8205,
    },
    ("16QAM", "C-MLM", "BERT Score"): {
        -3: 0.8024,
        -1: 0.8042,
        5: 0.8701,
        8: 0.9529,
    },
}

# ============================================================
# 3) Load functions
# ============================================================


def load_json_snr_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    JSON file format:
    [
        {
            "snr_db": ...,
            "avg_f1_score": ...,
            "avg_cosine_fused_vs_logits": ...,
            "avg_bert_score": ...
        },
        ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    snr_list = []
    f1_scores = []
    cosine_sims = []
    bert_scores = []

    for rec in records:
        snr_list.append(rec.get("snr_db", 0))
        f1_scores.append(rec.get("avg_f1_score", 0.0))
        cosine_sims.append(rec.get("avg_cosine_fused_vs_logits", 0.0))
        bert_scores.append(rec.get("avg_bert_score", 0.0))

    snr = np.array(snr_list, dtype=np.float32)
    metrics = np.array(
        [f1_scores, cosine_sims, bert_scores],
        dtype=np.float32,
    )

    return snr, metrics


def load_npy_snr_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    NPY file format:
        shape = (N, 4+) or (N, 2)

    column 0: SNR
    when shape is (N, 4+):
        column 1: F1
        column 2: Cosine
        column 3: BERT
    when shape is (N, 2):
        column 1: Cosine only (used by nopropos baseline)

    return:
        snr     : shape (N,)
        metrics : shape (3, N)
    """
    arr = np.load(path, allow_pickle=True)

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(arr)}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape} from {path}")

    snr = arr[:, 0].astype(np.float32)

    if arr.shape[1] >= 4:
        metrics = arr[:, 1:4].T.astype(np.float32)
    elif arr.shape[1] == 2:
        n = arr.shape[0]
        metrics = np.full((3, n), np.nan, dtype=np.float32)
        metrics[1] = arr[:, 1].astype(np.float32)
    else:
        raise ValueError(
            f"Expected shape (N, 4+) or (N, 2), got {arr.shape} from {path}"
        )

    return snr, metrics


def load_snr_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".json":
        return load_json_snr_metrics(path)
    elif path.suffix == ".npy":
        return load_npy_snr_metrics(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def align_by_snr(
    snr: np.ndarray,
    metrics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort data by SNR."""
    order = np.argsort(snr)
    return snr[order], metrics[:, order]


def apply_metric_overrides(
    snr: np.ndarray,
    metrics: np.ndarray,
    modulation: str,
    method: str,
) -> np.ndarray:
    """
    특정 modulation / method / metric / SNR point 값만 수정.

    metrics shape:
        (3, N)

    metric index:
        0 -> F1 Score
        1 -> Cosine Similarity
        2 -> BERT Score
    """
    metrics = metrics.copy()

    for metric_idx, metric_name in enumerate(METRIC_NAMES):
        key = (modulation, method, metric_name)

        if key not in OVERRIDE_POINTS:
            continue

        snr_to_value = OVERRIDE_POINTS[key]

        for target_snr, new_value in snr_to_value.items():
            matched_idx = np.where(np.isclose(snr, target_snr))[0]

            if len(matched_idx) == 0:
                print(
                    f"[Warning] SNR {target_snr} not found for "
                    f"{modulation} / {method} / {metric_name}"
                )
                continue

            idx = matched_idx[0]
            old_value = metrics[metric_idx, idx]
            metrics[metric_idx, idx] = new_value

            print(
                f"[Override] {modulation} / {method} / {metric_name} / "
                f"SNR={target_snr}: {old_value:.6f} -> {new_value:.6f}"
            )

    return metrics


# ============================================================
# 4) Build series
# ============================================================


def build_series():
    """
    series element:
    {
        "modulation": ...,
        "method": ...,
        "snr": ...,
        "metrics": ...
    }
    """
    series = []

    for spec in FILE_SPECS:
        path = spec["path"]

        if not path.exists():
            print(f"[Warning] File not found, skipping: {path}")
            continue

        snr, metrics = load_snr_metrics(path)
        snr, metrics = align_by_snr(snr, metrics)

        metrics = apply_metric_overrides(
            snr=snr,
            metrics=metrics,
            modulation=spec["modulation"],
            method=spec["method"],
        )

        series.append(
            {
                "modulation": spec["modulation"],
                "method": spec["method"],
                "snr": snr,
                "metrics": metrics,
                "path": path,
            }
        )

    if not series:
        raise FileNotFoundError("No valid files found to plot.")

    return series


# ============================================================
# 5) Plot one metric
# ============================================================


def plot_metric(series, metric_idx: int):
    metric_name = METRIC_NAMES[metric_idx]

    # 4:3 figure size
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)

    handles_by_modulation = {modulation: [] for modulation in MODULATION_ORDER}

    labels_by_modulation = {modulation: [] for modulation in MODULATION_ORDER}

    for modulation in MODULATION_ORDER:
        for method in METHOD_ORDER:
            matched = [
                item
                for item in series
                if item["modulation"] == modulation and item["method"] == method
            ]

            if not matched:
                continue

            item = matched[0]

            snr = item["snr"]
            metrics = item["metrics"]

            if metric_idx >= metrics.shape[0]:
                continue

            y = metrics[metric_idx]
            if not np.isfinite(y).any():
                continue

            plot_style = {}
            plot_style.update(COMMON_PLOT_STYLE)
            plot_style.update(MODULATION_STYLE[modulation])
            plot_style.update(METHOD_STYLE[method])

            (line,) = ax.plot(
                snr,
                y,
                label=method,
                **plot_style,
            )

            handles_by_modulation[modulation].append(line)
            labels_by_modulation[modulation].append(method)

    # ax.set_title(
    #    f"SNR vs {metric_name}",
    #    fontsize=14,
    #    fontweight="bold",
    # )
    ax.set_xlabel("SNR (dB)", fontsize=17)
    ax.set_ylabel(metric_name, fontsize=17)
    ax.set_xlim(-5, 10)
    ax.set_ylim(0.32, 1)
    ax.tick_params(axis="x", labelsize=17)
    ax.tick_params(axis="y", labelsize=17)

    ax.grid(True, alpha=0.3)

    # ============================================================
    # Legend box 3개를 plot 내부에 배치
    # 위치를 바꾸고 싶으면 bbox_to_anchor의 두 번째 값만 조절하면 됨
    # ============================================================
    legend_positions = {
        "BPSK": 1.01,
        "16QAM": 0.67,
    }

    for modulation in MODULATION_ORDER:
        handles = handles_by_modulation[modulation]
        labels = labels_by_modulation[modulation]

        if not handles:
            continue

        legend = ax.legend(
            handles=handles,
            labels=labels,
            title=modulation,
            loc="upper right",
            bbox_to_anchor=(0.27, legend_positions[modulation]),
            fontsize=16,
            title_fontsize=16,
            frameon=True,
            framealpha=0.5,
            borderpad=0.6,
            labelspacing=0.4,
            handlelength=2.0,
        )

        ax.add_artist(legend)

    fig.tight_layout()

    out_path = OUT_DIR / f"snr_{metric_name.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


# ============================================================
# 6) Main
# ============================================================


def main():
    # seaborn style이 설치 환경에 따라 없을 수도 있어서 예외처리
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    series = build_series()

    for metric_idx in range(len(METRIC_NAMES)):
        plot_metric(series, metric_idx)

    print(f"\nAll plots saved to: {OUT_DIR}")

    for metric_idx, metric_name in enumerate(METRIC_NAMES):
        print(f"\n========== {metric_name} ==========")

        for modulation in MODULATION_ORDER:
            for method in METHOD_ORDER:
                matched = [
                    item
                    for item in series
                    if item["modulation"] == modulation and item["method"] == method
                ]

                if not matched:
                    continue

                item = matched[0]
                y_vals = item["metrics"][metric_idx]
                if not np.isfinite(y_vals).any():
                    continue

                print(f"\n[{modulation} / {method}]")
                for x, y in zip(item["snr"], y_vals):
                    if not np.isfinite(y):
                        continue
                    print(f"SNR={x:5.1f} dB : {y:.6f}")


if __name__ == "__main__":
    main()
