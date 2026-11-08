"""Create a Korean result summary and publication-style plots from actual runs.

No sample scientific results are included. This module reads completed CSV
outputs and never fills missing scores with zeros or fabricated measurements.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ORDER = ["A_pretrained", "B_gamma", "B_CLAHE", "C_ZeroDCE", "D_raw", "E_always", "M_raw", "M_always", "M_selected"]
TRAINED = ["D_raw", "E_always", "M_raw", "M_always", "M_selected"]
LIGHT_ORDER = ["Low", "Ambient", "Object", "Single", "Weak", "Strong", "Screen", "Window", "Shadow", "Twilight"]
NAVY, TEAL, ORANGE, GRAY = "#18344E", "#137F86", "#D97832", "#8B98A7"
COLORS = {"D_raw": NAVY, "E_always": GRAY, "M_raw": "#4D7D91", "M_always": TEAL, "M_selected": ORANGE}
SHORT = {
    "A_pretrained": "A\nPretrained", "B_gamma": "B\nGamma", "B_CLAHE": "B\nCLAHE",
    "C_ZeroDCE": "C\nZero-DCE++", "D_raw": "D\nRaw", "E_always": "E\nAlways",
    "M_raw": "M\nRaw", "M_always": "M\nAlways", "M_selected": "M\nSelected",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _fmt(value, digits=2):
    value = _number(value)
    return "—" if value is None else f"{value:.{digits}f}"


def _percent(value, digits=2):
    value = _number(value)
    return "—" if value is None else f"{100 * value:.{digits}f}"


def _markdown_table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(clean, row)) + " |" for row in rows)
    return "\n".join(lines)


def _ordered(frame):
    ranks = {name: i for i, name in enumerate(ORDER)}
    return frame.assign(_rank=frame["experiment"].map(ranks).fillna(len(ORDER))).sort_values(["_rank", "experiment"]).drop(columns="_rank")


def _read_table(path, required, stage):
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} 파일이 없습니다. 먼저 {stage}를 실행하세요.")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}의 필수 열이 없습니다: {sorted(missing)}")
    for column in ("mAP50", "mAP50_95", "AR100", "threshold", "mean_ms", "median_ms", "p95_ms", "fps", "enhanced_ratio", "n_images", "repeats"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB5C0")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E1E7EC", linewidth=0.7)
    axis.tick_params(colors=NAVY, labelsize=9)


def _save(figure, target, note):
    figure.text(0.02, 0.012, note, color="#586878", fontsize=8)
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    figure.savefig(target, dpi=180, facecolor="white", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)
    return str(target)


def _make_plots(metrics, latency, sweep, protocol, directory):
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    selected = _number(protocol.get("selected_threshold"))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": NAVY,
                         "axes.titleweight": "bold", "axes.titlecolor": NAVY,
                         "figure.facecolor": "white", "axes.facecolor": "white"}):
        overall = _ordered(metrics.query("dataset == 'exdark_test' and group == 'all'"))
        figure, axis = plt.subplots(figsize=(11.4, 5.0))
        x = np.arange(len(overall))
        values = overall["mAP50"].to_numpy(dtype=float) * 100
        colors = [COLORS.get(name, "#BCC7D1") for name in overall["experiment"]]
        bars = axis.bar(x, values, color=colors, width=0.65)
        axis.set_xticks(x, [SHORT.get(name, name) for name in overall["experiment"]])
        axis.set_ylim(0, 105)
        axis.set_ylabel("mAP50 (%)")
        axis.set_title("ExDark test: object detection performance", loc="left", pad=17)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", fontsize=9, color=NAVY)
        _style_axis(axis)
        paths.append(_save(figure, directory / "1_exdark_map.png", "Official ExDark test split | COCO bbox AP50, maxDets=100 | Missing values are not imputed."))

        lights = metrics[(metrics["dataset"] == "exdark_test") & metrics["group"].str.startswith("light:")]
        present = set(lights["group"].str.replace("light:", "", regex=False))
        light_names = [n for n in LIGHT_ORDER if n in present] + sorted(present - set(LIGHT_ORDER))
        figure, axis = plt.subplots(figsize=(12.0, 5.0))
        x, width = np.arange(len(light_names)), 0.25
        for i, experiment in enumerate(("D_raw", "E_always", "M_selected")):
            subset = lights[lights["experiment"] == experiment].set_index("group")
            values = [float(subset.loc[f"light:{name}", "mAP50"]) * 100 if f"light:{name}" in subset.index else np.nan for name in light_names]
            axis.bar(x + (i - 1) * width, values, width, label=experiment, color=COLORS[experiment])
        axis.set_xticks(x, light_names, rotation=22, ha="right")
        axis.set_ylim(0, 105)
        axis.set_ylabel("mAP50 (%)")
        axis.set_title("ExDark test: performance by lighting condition", loc="left", pad=17)
        axis.legend(frameon=False, ncol=3, loc="upper right")
        _style_axis(axis)
        paths.append(_save(figure, directory / "2_per_light.png", "D: raw training | E: enhanced training | M: mixed training | Class support and sample counts differ between groups."))

        figure, (accuracy_axis, time_axis) = plt.subplots(1, 2, figsize=(12.0, 4.6))
        for group, color, label in (("all", NAVY, "Pooled"), ("lowlight", TEAL, "Low-light"), ("bright", ORANGE, "Bright")):
            subset = sweep[sweep["group"] == group].sort_values("threshold")
            accuracy_axis.plot(subset["threshold"], subset["mAP50"] * 100, marker="o", markersize=4.5, linewidth=1.6, color=color, label=label)
        time_rows = sweep[sweep["group"] == "all"].sort_values("threshold")
        time_axis.plot(time_rows["threshold"], time_rows["mean_ms"], marker="o", markersize=4.5, linewidth=1.6, color=NAVY)
        thresholds = sorted(sweep["threshold"].dropna().unique())
        ticks = ["0\nRaw" if t == 0 else "256\nAlways" if t == 256 else f"{t:g}" for t in thresholds]
        for axis in (accuracy_axis, time_axis):
            axis.set_xticks(thresholds, ticks)
            axis.set_xlabel("Brightness threshold (0–255 luminance scale)")
            if selected is not None:
                axis.axvline(selected, color=GRAY, linestyle="--", linewidth=1.1, zorder=0)
            _style_axis(axis)
        accuracy_axis.set_ylabel("mAP50 (%)")
        accuracy_axis.set_ylim(0, 105)
        accuracy_axis.set_title("Validation accuracy", loc="left")
        accuracy_axis.legend(frameon=False, fontsize=9, loc="lower right")
        time_axis.set_ylabel("Mean processing time (ms/image)")
        time_axis.set_ylim(bottom=0)
        time_axis.set_title("Validation processing time", loc="left")
        note = "Validation only; thresholds are fixed before test evaluation. 0 = no enhancement; 256 = always enhance."
        if selected is not None:
            note += f" Dashed line: selected {selected:g}."
        paths.append(_save(figure, directory / "3_threshold_validation.png", note))

        mixed = metrics.query("dataset == 'mixed_test' and group == 'all'")
        mixed = mixed[mixed["experiment"].isin(TRAINED)].merge(latency[["experiment", "mean_ms"]], on="experiment", how="inner")
        figure, axis = plt.subplots(figsize=(8.6, 5.4))
        offsets = {"D_raw": (-34, -20), "E_always": (8, 15), "M_raw": (8, -21), "M_always": (-60, 18), "M_selected": (9, 12)}
        valid_x, valid_y = [], []
        for _, row in mixed.iterrows():
            xv, yv = _number(row["mean_ms"]), _number(row["mAP50"])
            if xv is None or yv is None:
                continue
            yv *= 100
            experiment = row["experiment"]
            valid_x.append(xv)
            valid_y.append(yv)
            axis.scatter(xv, yv, s=135 if experiment == "M_selected" else 65,
                         marker="*" if experiment == "M_selected" else "o",
                         color=COLORS[experiment], edgecolors="white", linewidths=0.7, zorder=3)
            axis.annotate(experiment, (xv, yv), xytext=offsets.get(experiment, (8, 8)), textcoords="offset points",
                          fontsize=9, color=COLORS[experiment], arrowprops={"arrowstyle": "-", "color": "#CBD3DB", "lw": 0.7})
        if valid_x:
            spread_x = max(max(valid_x) - min(valid_x), max(valid_x) * 0.2, 0.5)
            spread_y = max(max(valid_y) - min(valid_y), 2.0)
            axis.set_xlim(max(0, min(valid_x) - 0.25 * spread_x), max(valid_x) + 0.3 * spread_x)
            axis.set_ylim(max(0, min(valid_y) - 0.35 * spread_y), min(100.5, max(valid_y) + 0.4 * spread_y))
        axis.set_xlabel("Mean processing time (ms/image)")
        axis.set_ylabel("Mixed-test mAP50 (%)")
        axis.set_title("Mixed test: accuracy and processing time", loc="left", pad=17)
        _style_axis(axis)
        axis.grid(axis="x", color="#E1E7EC", linewidth=0.7)
        paths.append(_save(figure, directory / "4_accuracy_latency.png", "Lower latency and higher mAP50 are preferable. Warm batch=1 pipeline; decoding and model loading are excluded."))
    return paths


def _lookup(frame, experiment, group="all", dataset="mixed_test"):
    matches = frame[(frame["experiment"] == experiment) & (frame["group"] == group) & (frame["dataset"] == dataset)]
    return matches.iloc[0].to_dict() if len(matches) else {}


def _comparisons(metrics, latency):
    lookup = latency.set_index("experiment").to_dict("index")
    selected = _lookup(metrics, "M_selected")
    paragraphs = []
    for reference in ("M_always", "D_raw"):
        baseline = _lookup(metrics, reference)
        ap_new, ap_old = _number(selected.get("mAP50")), _number(baseline.get("mAP50"))
        if ap_new is None or ap_old is None:
            continue
        text = f"**M_selected와 {reference} 비교:** 혼합 테스트 mAP50 변화는 **{100 * (ap_new - ap_old):+.2f}%p** "
        text += f"({_percent(ap_old)}% → {_percent(ap_new)}%)이다."
        new_ms = _number(lookup.get("M_selected", {}).get("mean_ms"))
        old_ms = _number(lookup.get(reference, {}).get("mean_ms"))
        if new_ms is not None and old_ms is not None and old_ms > 0:
            text += f" 평균 처리 시간 변화는 **{100 * (new_ms / old_ms - 1):+.2f}%** ({old_ms:.2f} → {new_ms:.2f} ms/장)이다. 시간 변화의 음수는 감소, 양수는 증가를 뜻한다."
        group_changes = []
        for group, korean in (("lowlight", "저조도 부분집합"), ("bright", "밝은 이미지 부분집합")):
            new_ap = _number(_lookup(metrics, "M_selected", group).get("mAP50"))
            old_ap = _number(_lookup(metrics, reference, group).get("mAP50"))
            if new_ap is not None and old_ap is not None:
                group_changes.append(f"{korean} {100 * (new_ap - old_ap):+.2f}%p")
        if group_changes:
            text += " 그룹별 mAP50 변화는 " + ", ".join(group_changes) + "이다."
        paragraphs.append(text)
    if not paragraphs:
        paragraphs.append("필수 비교 행에 정의된 mAP50 값이 없어 자동 차이를 계산하지 않았다.")
    return paragraphs


def make_report(run_path):
    """Return {'summary': absolute Markdown path, 'figures': absolute PNG paths}."""
    run = Path(run_path).resolve()
    results = run / "results"
    metrics = _read_table(results / "test_metrics.csv", ["experiment", "dataset", "group", "mAP50", "mAP50_95"], "lab.final_test()")
    latency = _read_table(results / "latency.csv", ["experiment", "mean_ms", "fps", "enhanced_ratio"], "lab.benchmark()")
    sweep = _read_table(results / "threshold_validation.csv", ["threshold", "group", "mAP50", "mean_ms"], "lab.tune()")
    if metrics.duplicated(["experiment", "dataset", "group"]).any() or latency.duplicated(["experiment"]).any() or sweep.duplicated(["threshold", "group"]).any():
        raise ValueError("결과 CSV에 중복 실험/그룹 행이 있습니다. 서로 다른 실행 결과를 섞지 마세요.")
    completed = set(zip(metrics["experiment"], metrics["dataset"], metrics["group"]))
    required = {(experiment, dataset, "all") for experiment in TRAINED for dataset in ("exdark_test", "mixed_test")}
    required.update((experiment, "mixed_test", group) for experiment in TRAINED for group in ("lowlight", "bright"))
    if not required.issubset(completed) or not set(TRAINED).issubset(set(latency["experiment"])):
        raise ValueError("D·E·M의 최종 평가/처리 시간 행이 아직 완성되지 않았습니다. lab.final_test()와 lab.benchmark()를 마친 뒤 실행하세요.")
    config, environment = _json(run / "config.json"), _json(run / "environment.json")
    protocol, training = _json(run / "protocol.json"), _json(run / "training_data.json")
    if config.get("mode") == "smoke":
        raise RuntimeError("smoke 결과는 최종 실험 보고서로 만들 수 없습니다. MODE='full' 실험을 완료하세요.")
    if "selected_threshold" not in protocol:
        raise FileNotFoundError("protocol.json의 확정 기준값이 없습니다. lab.tune()을 먼저 실행하세요.")
    figures = _make_plots(metrics, latency, sweep, protocol, results / "figures")
    threshold = float(protocol["selected_threshold"])
    tolerance = protocol.get("selection", {}).get("allowed_drop", config.get("allowable_mAP50_drop", 0.01))
    hardware_values = latency["hardware"].dropna().astype(str).unique().tolist() if "hardware" in latency else []
    hardware = ", ".join(hardware_values) or environment.get("gpu") or ("CPU" if environment.get("device") == "cpu" else "기록 없음")
    hardware_warning = "" if len(hardware_values) <= 1 else "\n\n**주의:** 처리 시간 표에 서로 다른 장치가 포함되어 있으므로 동일 장치에서 다시 비교해야 한다."
    sections = [
        "# 선택적 저조도 보정 객체 탐지 — 실험 결과",
        f"실행 이름: `{run.name}`. 아래 숫자는 저장된 실험 결과에서 계산했다. mAP는 0–1 값을 0–100으로 변환한 **% 단위**, mAP 차이는 **%p 단위**다. 일반적인 분류 정확도와 같은 지표는 아니다.",
        "입력 자료: [최종 평가 CSV](test_metrics.csv), [처리 시간 CSV](latency.csv), [검증 기준값 CSV](threshold_validation.csv).",
        "## 1. 실행 설정",
        _markdown_table(["항목", "기록"], [
            ["YOLO 모델", "YOLO11n, COCO 사전학습 가중치에서 시작"],
            ["보정 모델", "Zero-DCE++, 사전학습 가중치 고정"],
            ["학습 seed / epochs / batch", f"{config.get('seed', '미기록')} / {config.get('epochs', '미기록')} / {config.get('batch', '미기록')}"],
            ["탐지 입력 / 공통 전처리 최대 변", f"{config.get('imgsz', '미기록')} / {config.get('max_side', '미기록')} px"],
            ["모델별 학습 / 검증 이미지", f"{training.get('train', '미기록')} / {training.get('val', '미기록')}장"],
            ["M 보정본 수: 학습 / 검증", f"{training.get('M_enhanced_train', '미기록')} / {training.get('M_enhanced_val', '미기록')}장"],
            ["측정 장치", hardware],
            ["선택된 밝기 기준값", f"{threshold:g}"],
            ["검증셋 선택 허용 감소폭", f"mAP50 {100 * float(tolerance):g}%p, 전체·저조도·밝은 이미지 각 그룹"],
            ["검증셋 구성", protocol.get("selection_split", "protocol.json 참조")],
        ]) + hardware_warning,
        "M은 학습 원본 이미지마다 한 가지 버전을 배정하는 고정 50:50 혼합 방식이다. 원본과 보정본을 모두 추가해 학습 횟수를 두 배로 늘리지 않았으며, D·E·M의 이미지 수와 학습 설정을 맞췄다. M의 검증 입력 구성도 고정되어 있다.",
    ]
    if threshold == 0:
        sections.append("**선택 결과 해석:** 검증 절차에서 **보정 없음(기준값 0)**이 선택되었다. 이 실행은 중간 기준값을 사용하는 선택적 보정의 우수성을 입증한 결과로 해석하지 않는다.")
    elif threshold == 256:
        sections.append("**선택 결과 해석:** 검증 절차에서 **항상 보정(기준값 256)**이 선택되었다. 이 설정에서 보정을 일부 생략하는 정책의 이점이 확인되었다고 주장하지 않는다.")
    else:
        sections.append(f"**선택 결과 해석:** 공통 리사이즈 후 패딩 전 이미지의 PIL L 평균 밝기가 **{threshold:g} 미만**이면 보정한다. 기준값은 검증 결과로 정한 후 고정했으며, 최종 테스트 결과로 다시 선택하지 않는다.")
    overall = _ordered(metrics.query("dataset == 'exdark_test' and group == 'all'"))
    sections += ["## 2. ExDark 공식 테스트 분할", _markdown_table(
        ["실험", "이미지 수", "mAP50 (%)", "mAP50–95 (%)"],
        [[r["experiment"], _fmt(r.get("n_images"), 0), _percent(r["mAP50"]), _percent(r["mAP50_95"])] for _, r in overall.iterrows()]),
        "![ExDark 전체 결과](figures/1_exdark_map.png)"]
    mixed_rows = []
    for experiment in TRAINED:
        pooled = _lookup(metrics, experiment)
        if not pooled:
            continue
        mixed_rows.append([experiment, _percent(pooled.get("mAP50")), _percent(_lookup(metrics, experiment, "lowlight").get("mAP50")),
                           _percent(_lookup(metrics, experiment, "bright").get("mAP50")), _percent(pooled.get("mAP50_95"))])
    sections += ["## 3. 저조도·밝은 이미지 혼합 테스트", _markdown_table(
        ["실험", "전체 mAP50 (%)", "저조도 mAP50 (%)", "밝은 이미지 mAP50 (%)", "전체 mAP50–95 (%)"], mixed_rows),
        "전체 mAP는 혼합셋의 예측·정답을 함께 평가한 값이다. 두 그룹의 mAP를 단순 평균하지 않았다. ExDark 저조도 부분집합과 밝은 COCO val2017 부분집합의 출처 차이가 함께 포함된다.",
        "## 4. 실제 처리 시간", _markdown_table(
            ["실험", "평균 ms/장", "중앙값 ms", "P95 ms", "FPS", "보정 비율 (%)", "측정 이미지 × 반복"],
            [[r["experiment"], _fmt(r["mean_ms"]), _fmt(r.get("median_ms")), _fmt(r.get("p95_ms")), _fmt(r["fps"]), _percent(r["enhanced_ratio"]),
              f"{_fmt(r.get('n_images'), 0)} × {_fmt(r.get('repeats'), 0)}"] for _, r in _ordered(latency).iterrows()]),
        "평균 FPS는 1,000 ÷ 평균 ms/장으로 계산했다. 측정 범위는 RAM에 준비된 RGB 이미지부터 밝기 판단·필요시 실제 보정·리사이즈·색상 변환·장치 전송·YOLO 전처리·추론·NMS까지다. 파일 읽기·이미지 디코딩·모델 로딩·화면 출력·네트워크 전송은 제외했다. 동일 이미지 표본과 순서, batch=1, FP32 조건에서 예열 후 반복 측정했으며 CUDA에서는 경계마다 연산 완료를 기다렸다. 이 FPS를 웹앱이나 휴대전화의 실제 영상 처리 FPS로 해석하지 않는다.",
        "![정확도와 처리 시간](figures/4_accuracy_latency.png)",
        "## 5. 비교 결과 해석", *_comparisons(metrics, latency),
        "위 차이는 이 실행에서 관측한 값이다. 정확도 차이와 시간 차이는 서로 다른 단위이며, 단일 seed 결과만으로 통계적 유의성이나 다른 장치에서의 향상을 단정하지 않는다.",
        "## 6. 검증 기준값과 조명별 결과",
        "![검증셋 기준값 선택](figures/3_threshold_validation.png)",
        "기준값 그래프는 검증셋 결과다. 0은 무보정, 256은 항상 보정을 뜻한다. 회색 점선은 선택된 값이다. 전체·저조도·밝은 이미지 mAP50 감소폭을 제한한 후보 중 실제 검증 처리 시간이 가장 짧은 후보를 선택했다.",
        "![조명별 탐지 결과](figures/2_per_light.png)",
        "조명별 표본 수와 포함 클래스 수가 다르므로 작은 그룹의 점수 변화는 주의해서 해석한다. 보정이 실패한 원인을 특정하려면 정답 박스와 원본·보정 이미지, 예측 박스를 함께 확인해야 한다.",
        "## 7. 평가 방법과 재현 정보",
        f"정량 평가는 confidence={config.get('evaluation_conf', 0.001)}, NMS IoU={config.get('evaluation_nms_iou', 0.7)}, 최대 예측 300개를 사용한다. 최종 지표는 공식 COCO bbox 평가기의 maxDets=[1,10,100] 기준이다. 학습 화면의 Ultralytics 내부 mAP와 수치가 완전히 같을 필요는 없으며, 최종 방법 간 비교에는 이 보고서의 동일 평가기 결과를 사용한다. `AR100`은 여러 IoU 기준에서 평균한 COCO 재현율로, 단일 confidence에서의 재현율과 구분한다.",
        "데모의 박스 표시는 confidence=0.25를 사용한다. 정량 평가의 0.001과 목적이 다르므로 화면에 보이는 박스 수로 mAP 상승·하락을 판단하지 않는다.",
        "설정과 출처는 `config.json`, `environment.json`, `protocol.json`, `training_data.json`에 기록된다. `protocol.json`에는 선택 기준값과 모델 가중치 해시가 있다. 원래 CSV에는 반올림 전 0–1 지표가 보존된다. 하드웨어와 라이브러리 버전은 저장된 실제 기록을 사용한다.",
        "## 8. 해석 범위와 한계",
        "- 이 보고서는 한 실행 seed의 결과다. 학습 seed 반복 간 분산이나 신뢰구간을 추정하지 않았다. 시간 측정 반복은 학습 seed 반복을 대신하지 않는다.\n"
        "- ExDark와 COCO는 장면·객체 크기·클래스 구성·라벨링 기준도 다르다. 혼합 결과의 차이를 조도만의 인과 효과로 단정하지 않는다.\n"
        "- 밝은 이미지 선별은 평균 밝기 기준이며 실제 촬영 시각이 낮이라는 뜻은 아니다. 평균 밝기는 역광이나 일부만 어두운 객체를 충분히 설명하지 못한다.\n"
        "- ExDark의 Table을 COCO의 dining table에 대응시키는 등 클래스 의미 범위에 차이가 있다. 동일한 클래스 대응을 모든 실험에 적용했다.\n"
        "- M의 고정 50:50 원본·보정본 혼합 학습은 입력 표현에 대한 적응을 돕기 위한 설정이다. 밝은 COCO 장면에서의 성능을 자동으로 보장하지 않는다.\n"
        "- FPS는 기록된 장치와 구현·해상도·보정 비율에서의 측정치다. 다른 GPU·CPU·휴대전화에 그대로 적용하지 않는다.",
    ]
    target = results / "summary.md"
    target.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return {"summary": str(target), "figures": figures}
