"""Comparable COCO evaluation, online latency measurement, and validation selection.

The image_id used by this module is the 1-based position in ``records``. Never
combine predictions made from a different record order. AP uses the standard
COCO maximum of 100 detections per image; AR100 is COCO average recall, not
recall at a single confidence threshold. Accuracy and speed both default to
conf=0.001, IoU-NMS=0.7, max_det=300, FP32, batch=1.

No dataset images or model weights are embedded in this module.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


NAMES = [
    "bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup",
    "dog", "motorcycle", "person", "dining table",
]
EXDARK_TO_COCO = [1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]
_ALIASES = {"people": "person", "motorbike": "motorcycle", "table": "dining table"}


def class_mapping(names, require_all=True):
    """Map actual model output IDs to our 12 IDs using class names, not indices."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    mapping = {}
    for model_id, name in items:
        normalized = re.sub(r"\s+", " ", str(name).lower().replace("_", " ").strip())
        normalized = _ALIASES.get(normalized, normalized)
        if normalized in NAMES:
            mapping[int(model_id)] = NAMES.index(normalized)
    missing = sorted(set(range(len(NAMES))) - set(mapping.values()))
    if not mapping or (require_all and missing):
        labels = [NAMES[i] for i in missing]
        raise ValueError(
            f"Model class names do not match the project. Missing: {labels}. "
            "Use a COCO detector or a detector trained with this project's 12 classes."
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Multiple model class IDs map to the same target class.")
    return mapping


def records_signature(records):
    """Portable order-sensitive identity for the evaluation images and labels."""
    entries = []
    for rec in records:
        entries.append({
            key: rec.get(key) for key in
            ("id", "split", "source", "group", "light", "width", "height", "boxes", "boxes_crowd")
        })
    data = json.dumps(entries, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _check_records(records):
    if not records:
        raise ValueError("No records to evaluate. Check the manifest and split filters.")
    ids = [str(r["id"]) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs in evaluation records; do not repeat images to balance groups.")


def _read_rgb(record):
    with Image.open(record["path"]) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    expected = (int(record["height"]), int(record["width"]))
    if rgb.shape[:2] != expected:
        raise ValueError(
            f"Image/manifest dimensions disagree for {record['id']}: "
            f"file={rgb.shape[:2]}, manifest={expected}. Rebuild the manifest."
        )
    return rgb


def _load_model(model_or_path):
    if hasattr(model_or_path, "predict") and hasattr(model_or_path, "names"):
        return model_or_path
    from ultralytics import YOLO
    return YOLO(str(model_or_path))


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _predict(model, bgr_frames, mapping, imgsz, device, conf, iou, max_det):
    return model.predict(
        source=bgr_frames, imgsz=int(imgsz), device=device,
        conf=float(conf), iou=float(iou), max_det=int(max_det),
        classes=sorted(mapping), half=False, augment=False,
        verbose=False, save=False, stream=False,
    )


def collect_predictions(
    model_path, records, pipeline, imgsz=640, device="0", batch=1,
    conf=0.001, iou=0.7, max_det=300, progress=True,
):
    """Run the actual pipeline and return detections in original-image coordinates.

    ``pipeline`` receives RGB uint8 and returns (RGB uint8, enhanced_boolean).
    Ultralytics NumPy input is BGR, so conversion takes place before inference.
    Boxes are rescaled after any pipeline resize; the ground truth stays in
    the original image coordinate system.
    """
    records = list(records)
    _check_records(records)
    if int(batch) != batch or batch < 1:
        raise ValueError("batch must be a positive integer.")
    model = _load_model(model_path)
    mapping = class_mapping(model.names)
    predictions = []
    for start in range(0, len(records), int(batch)):
        chunk = records[start:start + int(batch)]
        frames, shapes = [], []
        for rec in chunk:
            transformed, _ = pipeline(_read_rgb(rec))
            transformed = np.asarray(transformed)
            if transformed.dtype != np.uint8 or transformed.ndim != 3 or transformed.shape[2] != 3:
                raise TypeError("The transformation pipeline must return RGB uint8 H×W×3.")
            shapes.append(transformed.shape[:2])
            frames.append(np.ascontiguousarray(transformed[:, :, ::-1]))
        results = _predict(model, frames, mapping, imgsz, device, conf, iou, max_det)
        if len(results) != len(chunk):
            raise RuntimeError("The number of prediction results differs from the number of input images.")
        for offset, (rec, result, shape) in enumerate(zip(chunk, results, shapes)):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = _to_numpy(result.boxes.xyxy)
            classes = _to_numpy(result.boxes.cls)
            scores = _to_numpy(result.boxes.conf)
            scale_x = float(rec["width"]) / shape[1]
            scale_y = float(rec["height"]) / shape[0]
            for box, model_cls, score in zip(xyxy, classes, scores):
                model_cls = int(model_cls)
                if model_cls not in mapping:
                    continue
                box = np.asarray(box, dtype=float) * np.array([scale_x, scale_y, scale_x, scale_y])
                box[[0, 2]] = np.clip(box[[0, 2]], 0, int(rec["width"]))
                box[[1, 3]] = np.clip(box[[1, 3]], 0, int(rec["height"]))
                x1, y1, x2, y2 = box.tolist()
                # Boxes clipped to the image edge can have zero area. Keep
                # them as non-matching detections rather than removing false
                # positives from the AP computation.
                if x2 < x1 or y2 < y1 or not np.isfinite(box).all() or not math.isfinite(float(score)):
                    raise ValueError(f"Invalid predicted box or score for {rec['id']}.")
                predictions.append({
                    "image_id": start + offset + 1,
                    "category_id": mapping[model_cls] + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })
        if progress and (start == 0 or (start + len(chunk)) % 100 == 0 or start + len(chunk) == len(records)):
            print(f"Inference {start + len(chunk)}/{len(records)}", flush=True)
    return predictions


def build_coco_ground_truth(records):
    """Create COCO ground truth without importing pycocotools (also useful in QA)."""
    records = list(records)
    _check_records(records)
    images, annotations = [], []
    for image_id, rec in enumerate(records, 1):
        width, height = int(rec["width"]), int(rec["height"])
        if width < 1 or height < 1:
            raise ValueError(f"Invalid image dimensions for {rec['id']}.")
        images.append({
            "id": image_id, "file_name": str(rec["id"]), "width": width, "height": height,
        })
        for field, iscrowd in (("boxes", 0), ("boxes_crowd", 1)):
            for row in rec.get(field, []):
                if len(row) != 5:
                    raise ValueError(f"Invalid ground-truth row for {rec['id']}: expected class,x1,y1,x2,y2.")
                cls, x1, y1, x2, y2 = row
                cls_int = int(cls)
                coords = [float(x1), float(y1), float(x2), float(y2)]
                if cls_int != cls or cls_int not in range(len(NAMES)) or not all(map(math.isfinite, coords)):
                    raise ValueError(f"Invalid ground-truth class/coordinates for {rec['id']}.")
                x1, y1, x2, y2 = coords
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > width + 1e-4 or y2 > height + 1e-4:
                    raise ValueError(f"Ground-truth box outside the image for {rec['id']}: {coords}.")
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id,
                    "category_id": cls_int + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": iscrowd,
                })
    return {
        "info": {"description": "12-class ExDark/COCO evaluation", "records_sha256": records_signature(records)},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(NAMES)],
    }


def _group_indices(records, include_lights):
    groups = {"all": list(range(1, len(records) + 1))}
    for group in sorted({str(r.get("group", r.get("source", "unknown"))) for r in records}):
        if group == "all":
            continue
        groups[group] = [i for i, r in enumerate(records, 1) if str(r.get("group", r.get("source", "unknown"))) == group]
    if include_lights:
        lights = sorted({str(r["light"]) for r in records if r.get("source") == "exdark" and r.get("light")})
        for light in lights:
            groups[f"light:{light}"] = [i for i, r in enumerate(records, 1) if r.get("source") == "exdark" and str(r.get("light")) == light]
    return groups


def evaluate_predictions(records, predictions, include_lights=True, include_classes=False):
    """Evaluate pooled predictions and groups directly with the official COCO API.

    Group mAPs are computed independently. The pooled mAP is NOT the arithmetic
    average of the lowlight and bright mAPs. A missing/undefined metric is None.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    records = list(records)
    predictions = list(predictions)
    ground_truth = build_coco_ground_truth(records)
    image_ids = set(range(1, len(records) + 1))
    for pred in predictions:
        if pred["image_id"] not in image_ids or pred["category_id"] not in range(1, len(NAMES) + 1):
            raise ValueError("Prediction IDs do not match this manifest's image/class IDs.")
        if len(pred["bbox"]) != 4 or not all(math.isfinite(float(v)) for v in pred["bbox"]):
            raise ValueError("Invalid prediction bounding box.")
        if pred["bbox"][2] < 0 or pred["bbox"][3] < 0 or not math.isfinite(float(pred["score"])):
            raise ValueError("Invalid prediction size or confidence.")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = ground_truth
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            # COCO.loadRes([]) indexes the first element and crashes.
            dt = COCO()
            dt.dataset = {**ground_truth, "annotations": []}
            dt.createIndex()
    groups = [(name, ids, list(range(1, len(NAMES) + 1))) for name, ids in _group_indices(records, include_lights).items()]
    if include_classes:
        groups.extend((f"class:{name}", sorted(image_ids), [i + 1]) for i, name in enumerate(NAMES))
    rows = []
    for group, ids, category_ids in groups:
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator = COCOeval(gt, dt, "bbox")
            evaluator.params.imgIds = ids
            evaluator.params.catIds = category_ids
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        stats = evaluator.stats
        def defined(index):
            value = float(stats[index])
            return value if math.isfinite(value) and value >= 0 else None
        id_set, cat_set = set(ids), set(category_ids)
        regular_gt = [ann for ann in ground_truth["annotations"] if ann["image_id"] in id_set and ann["category_id"] in cat_set and not ann["iscrowd"]]
        rows.append({
            "group": group, "n_images": len(ids), "n_gt_boxes": len(regular_gt),
            "n_classes_with_gt": len({a["category_id"] for a in regular_gt}),
            "mAP50": defined(1), "mAP50_95": defined(0),
            "AR100": defined(8), "recall100": defined(8),
            "AP_max_detections": 100,
        })
    return rows


def _pipeline_config(pipeline):
    result = {"type": type(pipeline).__name__}
    if callable(getattr(pipeline, "config", None)):
        declared = pipeline.config()
        if not isinstance(declared, dict):
            raise TypeError("pipeline.config() must return a JSON-serializable dictionary.")
        result.update(declared)
    for name in ("policy", "threshold", "max_side", "device", "weights", "gamma", "clip_limit"):
        if hasattr(pipeline, name):
            value = getattr(pipeline, name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[name] = value
            elif isinstance(value, Path):
                result[name] = str(value)
    return result


def _versions():
    versions = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "ultralytics", "pycocotools", "Pillow", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _model_metadata(model_path):
    result = {"model": str(model_path) if isinstance(model_path, (str, Path)) else type(model_path).__name__}
    if isinstance(model_path, (str, Path)) and Path(model_path).is_file():
        digest = hashlib.sha256()
        with Path(model_path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["weights_sha256"] = digest.hexdigest()
    return result


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def evaluate_model(
    model_path, records, pipeline, output_dir=None, run_name="evaluation", imgsz=640,
    device="0", batch=1, conf=0.001, iou=0.7, max_det=300,
    include_lights=True, include_classes=False, progress=True,
):
    """Run inference/evaluation, optionally saving CSV, predictions and protocol JSON."""
    records = list(records)
    predictions = collect_predictions(
        model_path, records, pipeline, imgsz=imgsz, device=device, batch=batch,
        conf=conf, iou=iou, max_det=max_det, progress=progress,
    )
    metrics = evaluate_predictions(records, predictions, include_lights, include_classes)
    splits = sorted({str(r.get("split", "unknown")) for r in records})
    protocol = {
        "run_name": run_name, "created_utc": datetime.now(timezone.utc).isoformat(),
        **_model_metadata(model_path), "pipeline": _pipeline_config(pipeline),
        "records_sha256": records_signature(records), "n_images": len(records),
        "split": splits[0] if len(splits) == 1 else "+".join(splits),
        "imgsz": int(imgsz), "batch": int(batch), "device": str(device),
        "conf": float(conf), "nms_iou": float(iou), "max_det": int(max_det),
        "half": False, "metric": "COCO bbox AP/AR, maxDets=[1,10,100]", "versions": _versions(),
        "images": [{"image_id": i, "id": r["id"], "source": r.get("source"), "group": r.get("group")} for i, r in enumerate(records, 1)],
    }
    for row in metrics:
        row.update({
            "run_name": run_name, "split": protocol["split"],
            "policy": protocol["pipeline"].get("policy", "unknown"),
            "threshold": protocol["pipeline"].get("threshold"),
            "records_sha256": protocol["records_sha256"],
        })
    files = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_name)).strip(".") or "evaluation"
        csv_path = output_dir / f"{stem}_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        prediction_path = output_dir / f"{stem}_predictions.json"
        protocol_path = output_dir / f"{stem}_protocol.json"
        _write_json(prediction_path, predictions)
        _write_json(protocol_path, protocol)
        files = {"metrics": str(csv_path), "predictions": str(prediction_path), "protocol": str(protocol_path)}
    return {"metrics": metrics, "predictions": predictions, "protocol": protocol, "files": files}


def stratified_sample(records, n=300, seed=2026):
    """Sample without replacement while approximately preserving domain proportions."""
    records = list(records)
    _check_records(records)
    if int(n) != n or n < 1:
        raise ValueError("n must be a positive integer.")
    n = min(int(n), len(records))
    by_group = defaultdict(list)
    for record in records:
        by_group[str(record.get("group", record.get("source", "unknown")))].append(record)
    targets = {g: n * len(items) / len(records) for g, items in by_group.items()}
    counts = {g: int(math.floor(target)) for g, target in targets.items()}
    remaining = n - sum(counts.values())
    priority = sorted(by_group, key=lambda g: (-(targets[g] - counts[g]), g))
    for group in priority[:remaining]:
        counts[group] += 1
    rng = random.Random(seed)
    sampled = []
    for group in sorted(by_group):
        sampled.extend(rng.sample(by_group[group], counts[group]))
    rng.shuffle(sampled)
    return sampled


def _synchronizer(device):
    label = str(device).lower()
    if label == "cpu":
        return lambda: None, "CPU"
    import torch
    if label == "mps":
        return torch.mps.synchronize, "Apple MPS"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a GPU runtime, or explicitly use device='cpu'.")
    if "," in label:
        raise ValueError("Latency benchmarking uses one device, e.g. device='0'.")
    index = int(label.split(":")[-1]) if label not in ("cuda", "none", "") else 0
    return lambda: torch.cuda.synchronize(index), torch.cuda.get_device_name(index)


def benchmark_model(
    model_path, records, pipeline, imgsz=640, device="0", n=300,
    warmup=10, repeats=3, seed=2026, conf=0.001, iou=0.7, max_det=300,
    output_path=None, progress=True,
):
    """Measure warm per-image wall latency of online processing, including enhancement.

    Model loading and image-file reading/decoding are excluded. The same decoded
    RGB input arrays are prepared before timing. Timed work includes pipeline
    decisions/enhancement, resize, RGB/BGR conversion, model preprocessing,
    transfers, inference, and NMS. CUDA work is synchronized at both boundaries.
    This is serial batch=1 throughput, not a video decoder/web-app FPS claim.
    """
    if int(warmup) != warmup or warmup < 1 or int(repeats) != repeats or repeats < 1:
        raise ValueError("warmup and repeats must be positive integers.")
    selected = stratified_sample(records, n=n, seed=seed)
    frames = [_read_rgb(rec) for rec in selected]
    # Pipeline code must not modify its input in place; repeated timings need
    # the same raw input. Read-only arrays expose accidental modifications.
    for frame in frames:
        frame.flags.writeable = False
    model = _load_model(model_path)
    mapping = class_mapping(model.names)
    synchronize, hardware = _synchronizer(device)

    def process(rgb):
        transformed, enhanced = pipeline(rgb)
        transformed = np.asarray(transformed)
        if transformed.dtype != np.uint8 or transformed.ndim != 3 or transformed.shape[2] != 3:
            raise TypeError("The transformation pipeline must return RGB uint8 H×W×3.")
        bgr = np.ascontiguousarray(transformed[:, :, ::-1])
        _predict(model, [bgr], mapping, imgsz, device, conf, iou, max_det)
        return bool(enhanced)

    for index in range(int(warmup)):
        process(frames[index % len(frames)])
    synchronize()
    timings, enhanced_flags = [], []
    for repeat in range(int(repeats)):
        for frame in frames:
            synchronize()
            start = time.perf_counter()
            enhanced = process(frame)
            synchronize()
            timings.append((time.perf_counter() - start) * 1000)
            enhanced_flags.append(enhanced)
        if progress:
            print(f"Timing pass {repeat + 1}/{repeats}: {len(frames)} images", flush=True)
    values = np.asarray(timings, dtype=float)
    mean_ms = float(values.mean())
    result = {
        **_model_metadata(model_path), "pipeline": _pipeline_config(pipeline),
        "hardware": hardware, "device": str(device), "imgsz": int(imgsz),
        "batch": 1, "half": False, "conf": float(conf), "nms_iou": float(iou), "max_det": int(max_det),
        "n_images": len(selected), "repeats": int(repeats), "warmup": int(warmup),
        "n_measurements": len(values), "seed": int(seed),
        "mean_ms": mean_ms, "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)), "std_ms": float(values.std()),
        "fps": 1000.0 / mean_ms, "enhanced_ratio": float(np.mean(enhanced_flags)),
        "records_sha256": records_signature(selected), "sample_ids": [r["id"] for r in selected],
        "group_counts": dict((g, sum(str(r.get("group", r.get("source", "unknown"))) == g for r in selected)) for g in sorted({str(r.get("group", r.get("source", "unknown"))) for r in selected})),
        "timing_scope": "decoded RGB in RAM -> selection/enhancement/resize -> YOLO preprocessing/transfers/inference/NMS; batch=1; file I/O, image decoding and model load excluded",
        "timings_ms": timings, "versions": _versions(),
    }
    if output_path is not None:
        _write_json(output_path, result)
    return result


def select_threshold(rows, allowed_drop=0.01, reference_threshold=256, split="val"):
    """Choose the lowest measured latency subject to validation mAP50 constraints.

    Each flat row requires threshold, group, mAP50. An 'all' row for each
    threshold also requires mean_ms from the SAME warm batch=1 benchmark.
    Pass only the same model and validation dataset. Test rows are rejected.
    Scores use the 0..1 scale: allowed_drop=0.01 means one percentage point.
    Threshold 0 (no enhancement) is allowed to win; improvement is not assumed.
    """
    if split not in ("val", "validation", "mixed_val"):
        raise ValueError("Thresholds must be selected on validation data, never on the test set.")
    if not math.isfinite(float(allowed_drop)) or not 0 <= allowed_drop <= 1:
        raise ValueError("allowed_drop must be on the 0..1 scale.")
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    rows = list(rows)
    table = defaultdict(dict)
    signatures = set()
    for row in rows:
        row_split = row.get("split", split)
        if row_split not in ("val", "validation", "mixed_val"):
            raise ValueError(f"Found non-validation rows ({row_split}); do not tune on test results.")
        threshold = float(row["threshold"])
        if not math.isfinite(threshold) or not 0 <= threshold <= 256:
            raise ValueError("Brightness threshold must be in [0,256].")
        group = str(row["group"])
        if group in table[threshold]:
            raise ValueError(f"Duplicate threshold/group rows: {threshold}/{group}. Aggregate repeated seeds first.")
        table[threshold][group] = row
        if row.get("records_sha256"):
            signatures.add(str(row["records_sha256"]))
    if len(signatures) > 1:
        raise ValueError("Threshold comparisons use different evaluation records or record order.")
    reference = table.get(float(reference_threshold))
    if reference is None or "all" not in reference:
        raise ValueError("The always-enhance reference (threshold=256) must have an 'all' validation row.")
    constraint_groups = [g for g in ("all", "lowlight", "bright") if g in reference]
    reference_scores = {}
    for group in constraint_groups:
        value = reference[group].get("mAP50")
        if value is None or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"Undefined or out-of-range reference mAP50 for {group}.")
        reference_scores[group] = float(value)
    candidates = []
    for threshold, groups in sorted(table.items()):
        reasons, scores, drops = [], {}, {}
        for group in constraint_groups:
            row = groups.get(group)
            score = row.get("mAP50") if row is not None else None
            if score is None or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                reasons.append(f"missing/invalid {group} mAP50")
                continue
            scores[group] = float(score)
            drops[group] = reference_scores[group] - float(score)
            if drops[group] > float(allowed_drop) + 1e-12:
                reasons.append(f"{group} drop exceeds {allowed_drop}")
        mean_ms = groups.get("all", {}).get("mean_ms")
        if mean_ms is None or not math.isfinite(float(mean_ms)) or float(mean_ms) <= 0:
            reasons.append("missing/invalid measured mean_ms")
            mean_ms = None
        else:
            mean_ms = float(mean_ms)
        candidates.append({
            "threshold": int(threshold) if threshold.is_integer() else threshold,
            "feasible": not reasons, "mean_ms": mean_ms,
            "mAP50_by_group": scores, "drop_by_group": drops, "reasons": reasons,
        })
    feasible = [c for c in candidates if c["feasible"]]
    if not feasible:
        raise ValueError("No threshold has valid measured latency and all required accuracy constraints.")
    selected = min(feasible, key=lambda c: (c["mean_ms"], -c["mAP50_by_group"]["all"], c["threshold"]))
    return {
        "threshold": selected["threshold"], "selection_split": split,
        "metric": "mAP50", "allowed_drop": float(allowed_drop),
        "reference_threshold": reference_threshold, "reference_mAP50": reference_scores,
        "constraint_groups": constraint_groups, "selected": selected,
        "candidates": candidates, "records_sha256": next(iter(signatures), None),
        "selection_rule": "Lowest measured validation mean_ms with mAP50 loss <= allowed_drop in all, lowlight, and bright groups when present. Lock before test evaluation.",
    }
