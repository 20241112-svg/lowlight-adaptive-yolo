"""Notebook-facing workflow. All test evaluation is behind a saved protocol.

No scientific results are bundled. Synthetic smoke runs live in their own run
directory and cannot be exported as the final experiment.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .transforms import TransformPipeline, setup_zerodce, sha256_file


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timing_summary(timing):
    """Keep per-image IDs and timing samples private; publish aggregate metrics."""
    keep = ["mean_ms", "median_ms", "p95_ms", "std_ms", "fps", "enhanced_ratio",
            "n_images", "n_measurements", "repeats", "warmup", "batch", "imgsz",
            "conf", "nms_iou", "max_det", "device", "hardware", "timing_scope"]
    return {key: timing[key] for key in keep if key in timing}


def stable_sample(records, number, seed=2026):
    records = sorted(records, key=lambda r: r["id"])
    return random.Random(seed).sample(records, min(number, len(records)))


def manifest_digest(records):
    # Include label geometry as well as IDs. No raw data appears in public exports.
    compact = [{k: row.get(k) for k in ("id", "split", "width", "height", "boxes", "boxes_crowd")}
               for row in sorted(records, key=lambda r: r["id"])]
    return hashlib.sha256(json.dumps(compact, sort_keys=True).encode()).hexdigest()


def _link_or_copy(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


class Lab:
    def __init__(self, storage_root, data_root="/content/lowlight_data", mode="smoke",
                 tag="v1", epochs=30, batch=16, seed=0, imgsz=640, max_side=640):
        if mode not in {"smoke", "full"}:
            raise ValueError("MODE는 'smoke' 또는 'full'입니다.")
        if not tag.replace("_", "").replace("-", "").isalnum():
            raise ValueError("RUN_TAG에는 영문/숫자/-/_만 사용하세요.")
        self.mode, self.tag, self.seed = mode, tag, int(seed)
        self.epochs = 2 if mode == "smoke" else int(epochs)
        self.batch = min(8, int(batch)) if mode == "smoke" else int(batch)
        self.imgsz, self.max_side = int(imgsz), int(max_side)
        self.run_id = f"{tag}_{mode}_s{seed}_e{self.epochs}_b{self.batch}_i{imgsz}_p{max_side}"
        self.storage = Path(storage_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.run = self.storage / "experiments" / self.run_id
        self.results = self.run / "results"
        self.work = self.data_root / self.run_id
        for path in [self.storage, self.data_root, self.run, self.results, self.work]:
            path.mkdir(parents=True, exist_ok=True)
        self.exdark = []
        self.bright = []
        self.weights = None
        self.device = "0"
        config = {"mode": mode, "tag": tag, "epochs": self.epochs, "batch": self.batch,
                  "seed": seed, "imgsz": imgsz, "max_side": max_side,
                  "mixed_training": "fixed 50:50 raw/enhanced, one view per source image",
                  "threshold_candidates": [0, 30, 60, 90, 120, 256],
                  "allowable_mAP50_drop": 0.01,
                  "COCO_bright_threshold": 90,
                  "COCO_split_seed": 2026,
                  "evaluation_conf": 0.001, "evaluation_nms_iou": 0.7}
        config_file = self.run / "config.json"
        if config_file.exists() and read_json(config_file) != config:
            raise RuntimeError("같은 실행 이름의 설정이 다릅니다. RUN_TAG를 바꿔 새 실험으로 시작하세요.")
        write_json(config_file, config)
        print(f"실행: {self.run_id}\n결과 폴더: {self.run}")
        print("작동 확인용: train 120장/val 60장, 2 epoch; 최종 테스트 잠김." if mode == "smoke"
              else "본 실험: 공식 학습/검증 분할, 테스트는 기준값 확정 후에만 평가.")

    def setup(self, allow_cpu=False):
        import torch
        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError("GPU가 없습니다. Colab 런타임 → 런타임 유형 변경 → GPU를 선택하고 0단계를 다시 실행하세요.")
        self.device = "0" if torch.cuda.is_available() else "cpu"
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (작동 확인 전용)")
        self.weights = setup_zerodce(self.storage / "models" / "zerodce")
        env = {"python": sys.version, "platform": platform.platform(), "device": self.device,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "packages": {}, "zerodce": read_json(self.weights.parent / "provenance.json")}
        for package in ["torch", "torchvision", "ultralytics", "numpy", "opencv-python",
                        "Pillow", "pycocotools", "pandas", "matplotlib"]:
            try:
                env["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                env["packages"][package] = "missing"
        try:
            env["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            env["git_commit"] = "local-upload"
        write_json(self.run / "environment.json", env)
        self._base_model()
        print("환경 준비 완료. Zero-DCE++와 YOLO11n 가중치를 확보했습니다.")
        return env

    def _base_model(self):
        from ultralytics import YOLO
        target = self.storage / "models" / "yolo11n.pt"
        if not target.is_file():
            model = YOLO("yolo11n.pt")
            source = Path(getattr(model, "ckpt_path", "yolo11n.pt")).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return target

    def pipeline(self, policy, threshold=60):
        if self.weights is None:
            raise RuntimeError("lab.setup()을 먼저 실행하세요.")
        return TransformPipeline(policy=policy, threshold=threshold, weights=self.weights,
                                 device=self.device, max_side=self.max_side, scale_factor=12)

    def _raw_restore(self, name):
        cache = self.storage / "raw" / name
        local = self.data_root / "raw" / name
        local.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            for archive in cache.iterdir():
                if archive.is_file() and archive.suffix.lower() in {".zip", ".txt"}:
                    if not (local / archive.name).exists():
                        print("Drive에서 복구:", archive.name)
                        shutil.copy2(archive, local / archive.name)
        return local

    def _raw_backup(self, name, local):
        cache = self.storage / "raw" / name
        cache.mkdir(parents=True, exist_ok=True)
        for archive in Path(local).glob("*"):
            if archive.is_file() and archive.suffix.lower() in {".zip", ".txt"}:
                if archive.suffix.lower() == ".zip" and not zipfile.is_zipfile(archive):
                    continue
                target = cache / archive.name
                if not target.exists() or target.stat().st_size != archive.stat().st_size:
                    print("Drive 원자료 백업:", archive.name)
                    shutil.copy2(archive, target)

    def prepare_data(self):
        from .data import prepare_exdark
        raw = self._raw_restore("exdark")
        try:
            self.exdark = prepare_exdark(raw, self.data_root / "exdark_original")
        finally:
            self._raw_backup("exdark", raw)
        counts = Counter(row["split"] for row in self.exdark)
        print("ExDark 분할:", dict(counts))
        write_json(self.run / "data_audit.json", {"counts": dict(counts), "manifest_sha256": manifest_digest(self.exdark)})
        self._assert_splits(self.exdark)
        return dict(counts)

    @staticmethod
    def _assert_splits(records):
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError("중복 이미지 ID가 있습니다. 학습/검증/테스트 분할을 확인하세요.")

    def _trainval(self):
        if not self.exdark:
            raise RuntimeError("lab.prepare_data()를 먼저 실행하세요.")
        train = [r for r in self.exdark if r["split"] == "train"]
        val = [r for r in self.exdark if r["split"] == "val"]
        if self.mode == "smoke":
            train, val = stable_sample(train, 120, self.seed), stable_sample(val, 60, self.seed)
        return train, val

    def show_labels(self, n=6):
        import matplotlib.pyplot as plt
        from .data import NAMES
        train, _ = self._trainval()
        samples = stable_sample(train, n, self.seed)
        fig, axes = plt.subplots((len(samples) + 2) // 3, 3, figsize=(15, 4 * ((len(samples) + 2) // 3)), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat, samples):
            with Image.open(row["path"]) as original:
                image = original.convert("RGB")
            draw = ImageDraw.Draw(image)
            for class_id, x1, y1, x2, y2 in row["boxes"]:
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=max(2, image.width // 300))
                draw.text((x1, max(0, y1 - 12)), NAMES[int(class_id)], fill="yellow")
            axis.imshow(image)
            axis.set_title(row["light"])
        plt.tight_layout()
        return fig

    def prepare_views(self):
        from .data import NAMES
        from tqdm.auto import tqdm
        train, val = self._trainval()
        records = train + val
        fingerprint = {"data": manifest_digest(records), "max_side": self.max_side,
                       "zerodce": sha256_file(self.weights), "scale_factor": 12, "mix_seed": self.seed}
        marker = self.work / "views_complete.json"
        pending = self.work / "views_in_progress.json"
        for previous in [marker, pending]:
            if previous.exists() and read_json(previous) != fingerprint:
                raise RuntimeError("이미 준비한 학습 입력의 설정/원자료가 달라졌습니다. RUN_TAG를 바꿔 새 실험으로 시작하세요.")
        if marker.exists() and read_json(marker) == fingerprint:
            print("D/E/M 학습 이미지가 이미 준비되어 있습니다.")
            return
        write_json(pending, fingerprint)
        raw_pipe, enhanced_pipe = self.pipeline("none"), self.pipeline("always")
        enhanced_ids = set()
        for split_rows in [train, val]:
            chosen = stable_sample(split_rows, len(split_rows) // 2, self.seed)
            enhanced_ids.update(r["id"] for r in chosen)
        for row in tqdm(records, desc="D/E/M 학습 입력 준비"):
            with Image.open(row["path"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            stem, split = row["id"], row["split"]
            label_lines = []
            for class_id, x1, y1, x2, y2 in row["boxes"]:
                w, h = row["width"], row["height"]
                label_lines.append(f"{int(class_id)} {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
            for name, pipeline in [("D", raw_pipe), ("E", enhanced_pipe)]:
                target = self.work / name / "images" / split / (stem + ".png")
                label = self.work / name / "labels" / split / (stem + ".txt")
                target.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                if not target.is_file():
                    transformed, _ = pipeline(rgb)
                    Image.fromarray(transformed).save(target)
                label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            source_name = "E" if stem in enhanced_ids else "D"
            for folder, suffix in [("images", ".png"), ("labels", ".txt")]:
                _link_or_copy(self.work / source_name / folder / split / (stem + suffix),
                              self.work / "M" / folder / split / (stem + suffix))
        for name in ["D", "E", "M"]:
            dataset = {"path": str(self.work / name), "train": "images/train", "val": "images/val", "names": NAMES}
            (self.work / name / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
        write_json(marker, fingerprint)
        pending.unlink(missing_ok=True)
        write_json(self.run / "training_data.json", {"train": len(train), "val": len(val),
                   "M_enhanced_train": sum(r["id"] in enhanced_ids for r in train),
                   "M_enhanced_val": sum(r["id"] in enhanced_ids for r in val), **fingerprint})
        print(f"입력 준비 완료: 모델별 train {len(train)}, val {len(val)}. M의 원본/보정 비율 1:1.")

    def train(self, name):
        from ultralytics import YOLO
        if name not in {"D", "E", "M"}:
            raise ValueError("실험 이름은 D, E, M입니다.")
        dataset = self.work / name / "dataset.yaml"
        if not dataset.is_file():
            raise RuntimeError("lab.prepare_views()를 먼저 실행하세요.")
        target = self.run / "runs" / name
        best, last = target / "weights/best.pt", target / "weights/last.pt"
        marker = target / "completed.json"
        if marker.exists() and best.is_file():
            print(f"{name}: 이미 완료했습니다. {best}")
            return str(best)
        if last.is_file():
            # If the final epoch completed before the notebook wrote its marker,
            # Ultralytics has already stripped the optimizer. Do not resume it.
            history = target / "results.csv"
            if history.exists():
                with history.open() as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) >= self.epochs and best.is_file():
                    write_json(marker, {"epochs": self.epochs, "best_sha256": sha256_file(best)})
                    print(f"{name}: 마지막 epoch까지 완료된 결과를 복구했습니다.")
                    return str(best)
            print(f"{name}: last.pt부터 자동으로 이어서 학습합니다.")
            model = YOLO(str(last))
            model.train(resume=True, device=self.device)
        else:
            model = YOLO(str(self._base_model()))
            model.train(data=str(dataset), epochs=self.epochs, imgsz=self.imgsz,
                        batch=self.batch, seed=self.seed, deterministic=True,
                        device=self.device, workers=2, project=str(self.run / "runs"),
                        name=name, exist_ok=True, pretrained=True, optimizer="AdamW",
                        lr0=0.001, patience=self.epochs + 1, save=True, save_period=-1,
                        cache=False, plots=False, amp=self.device != "cpu",
                        close_mosaic=0 if self.mode == "smoke" else min(10, self.epochs),
                        verbose=True)
        if not best.is_file():
            raise RuntimeError(f"학습 완료 파일이 없습니다: {best}")
        write_json(marker, {"epochs": self.epochs, "best_sha256": sha256_file(best)})
        print(f"{name} 학습 완료: {best}")
        return str(best)

    def best(self, name):
        path = self.run / "runs" / name / "weights/best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"{name} 모델을 먼저 학습하세요: lab.train('{name}')")
        return path

    def baselines(self):
        from .evaluate import evaluate_model
        _, val = self._trainval()
        val = stable_sample(val, 60 if self.mode == "smoke" else 300, self.seed)
        rows = []
        for label, policy in [("A_pretrained", "none"), ("B_gamma", "gamma"),
                              ("B_CLAHE", "clahe"), ("C_ZeroDCE", "always")]:
            print(label)
            result = evaluate_model(str(self._base_model()), val, self.pipeline(policy),
                                    imgsz=self.imgsz, device=self.device)
            for row in result["metrics"]:
                rows.append({"experiment": label, "dataset": "validation_pilot", **row})
        write_csv(self.results / "baseline_validation.csv", rows)
        import pandas as pd
        return pd.DataFrame(rows).query("group == 'all'")

    def prepare_bright(self):
        if self.mode == "smoke":
            print("작동 확인 단계에서는 COCO 다운로드를 생략합니다. MODE='full'에서 실행하세요.")
            return []
        from .data import prepare_coco_bright
        raw = self._raw_restore("coco")
        try:
            self.bright = prepare_coco_bright(raw, self.data_root / "coco_bright", n_val=300, n_test=500,
                                              brightness_min=90, seed=2026)
        finally:
            self._raw_backup("coco", raw)
        self._assert_splits(self.exdark + self.bright)
        counts = Counter(r["split"] for r in self.bright)
        print("COCO 밝은 이미지 분할:", dict(counts))
        write_json(self.run / "bright_audit.json", {"counts": dict(counts), "manifest_sha256": manifest_digest(self.bright)})
        return dict(counts)

    def tune(self):
        if self.mode != "full":
            print("작동 확인 완료 후 MODE='full'로 바꿔 본 실험을 진행하세요. 기준값/최종 평가는 본 실험에만 저장합니다.")
            return None
        if (self.run / "test_complete.json").exists() or (self.run / "test_started.json").exists():
            raise RuntimeError("최종 테스트를 이미 확인했습니다. 같은 실험에서 기준값을 다시 고르지 마세요. 저장된 protocol.json을 사용하세요.")
        from .evaluate import evaluate_model, benchmark_model, select_threshold
        if not self.bright:
            raise RuntimeError("lab.prepare_bright()를 먼저 실행하세요.")
        dark_val = stable_sample([r for r in self.exdark if r["split"] == "val"], 300, 2026)
        bright_val = [r for r in self.bright if r["split"] == "val"]
        records = dark_val + bright_val
        self._assert_splits(records)
        rows = []
        timing_rows = []
        for threshold in [0, 30, 60, 90, 120, 256]:
            policy = "none" if threshold == 0 else "always" if threshold == 256 else "adaptive"
            pipeline = self.pipeline(policy, threshold)
            print("검증 기준값:", threshold)
            result = evaluate_model(str(self.best("M")), records, pipeline, imgsz=self.imgsz, device=self.device)
            timing = benchmark_model(str(self.best("M")), records, pipeline, imgsz=self.imgsz,
                                     device=self.device, n=120, repeats=3, warmup=10, seed=2026)
            timing_rows.append({"threshold": threshold, **timing_summary(timing)})
            write_json(self.results / "private" / f"validation_t{threshold}_timing.json", timing)
            for metric in result["metrics"]:
                if metric["group"] in {"all", "lowlight", "bright"}:
                    rows.append({"threshold": threshold, **metric, "mean_ms": timing["mean_ms"],
                                 "fps": timing["fps"], "enhanced_ratio": timing["enhanced_ratio"]})
            write_csv(self.results / "threshold_validation.csv", rows)
            write_csv(self.results / "threshold_latency.csv", timing_rows)
        selection = select_threshold(rows, allowed_drop=0.01)
        selected = float(selection["threshold"])
        protocol = {"selected_threshold": selected, "selection": selection,
                    "selection_split": "validation only: 300 ExDark val + 300 COCO bright val",
                    "validation_sha256": manifest_digest(records),
                    "weights_sha256": {name: sha256_file(self.best(name)) for name in ["D", "E", "M"]},
                    "config": read_json(self.run / "config.json"),
                    "pipeline": self.pipeline("adaptive", selected).config()}
        write_json(self.run / "protocol.json", protocol)
        print("최종 기준값 고정:", selected)
        print("0이면 보정 없음, 256이면 항상 보정이 선택된 것입니다. 이 결과도 그대로 보고합니다.")
        return protocol

    @staticmethod
    def _subset_metrics(all_records, predictions, subset):
        from .evaluate import evaluate_predictions
        index = {r["id"]: i + 1 for i, r in enumerate(all_records)}
        remap = {index[r["id"]]: j + 1 for j, r in enumerate(subset)}
        subset_predictions = [dict(p, image_id=remap[p["image_id"]]) for p in predictions if p["image_id"] in remap]
        return evaluate_predictions(subset, subset_predictions)

    def final_test(self, include_pretrained=True):
        if self.mode != "full":
            print("smoke 실행은 최종 테스트를 하지 않습니다. 본 실험 단계에서 실행하세요.")
            return None
        from .evaluate import collect_predictions, records_signature
        if not (self.run / "protocol.json").is_file():
            raise RuntimeError("lab.tune()으로 검증셋 기준값을 먼저 확정하세요.")
        if not self.exdark or not self.bright:
            raise RuntimeError("데이터를 복구한 뒤 실행하세요: prepare_data(), prepare_bright().")
        protocol = read_json(self.run / "protocol.json")
        for name, digest in protocol["weights_sha256"].items():
            if sha256_file(self.best(name)) != digest:
                raise RuntimeError("기준값 확정 이후 모델 가중치가 바뀌었습니다.")
        write_json(self.run / "test_started.json", {"protocol_sha256": sha256_file(self.run / "protocol.json")})
        dark = [r for r in self.exdark if r["split"] == "test"]
        bright = [r for r in self.bright if r["split"] == "test"]
        all_records = dark + bright
        mixed = stable_sample(dark, len(bright), 2026) + bright
        threshold = protocol["selected_threshold"]
        selected_policy = "none" if threshold == 0 else "always" if threshold == 256 else "adaptive"
        experiments = [("D_raw", self.best("D"), "none", 0),
                       ("E_always", self.best("E"), "always", 256),
                       ("M_raw", self.best("M"), "none", 0),
                       ("M_always", self.best("M"), "always", 256),
                       ("M_selected", self.best("M"), selected_policy, threshold)]
        if include_pretrained:
            experiments = [("A_pretrained", self._base_model(), "none", 0),
                           ("B_gamma", self._base_model(), "gamma", 0),
                           ("B_CLAHE", self._base_model(), "clahe", 0),
                           ("C_ZeroDCE", self._base_model(), "always", 256)] + experiments
        rows = []
        for name, model, policy, thr in experiments:
            cache = self.results / "private" / (name + "_predictions.json")
            key = {"dataset": records_signature(all_records), "weights": sha256_file(model),
                   "policy": policy, "threshold": thr, "imgsz": self.imgsz, "max_side": self.max_side}
            metadata = cache.with_suffix(".meta.json")
            if cache.is_file() and metadata.is_file() and read_json(metadata) == key:
                print("저장된 예측 재사용:", name)
                predictions = read_json(cache)
            else:
                print("최종 평가:", name)
                predictions = collect_predictions(str(model), all_records, self.pipeline(policy, thr),
                                                  imgsz=self.imgsz, device=self.device, batch=1)
                write_json(cache, predictions)
                write_json(metadata, key)
            for dataset, subset in [("exdark_test", dark), ("mixed_test", mixed)]:
                metrics = self._subset_metrics(all_records, predictions, subset)
                rows.extend({"experiment": name, "dataset": dataset, **m} for m in metrics)
            write_csv(self.results / "test_metrics.csv", rows)
        write_json(self.run / "test_complete.json", {"dataset_sha256": records_signature(all_records),
                   "mixed_sha256": records_signature(mixed), "experiments": [r[0] for r in experiments]})
        import pandas as pd
        return pd.DataFrame(rows).query("group in ['all','lowlight','bright']")

    def benchmark(self):
        if self.mode != "full":
            print("최종 처리 시간 측정은 본 실험에서 실행합니다.")
            return None
        from .evaluate import benchmark_model
        protocol = read_json(self.run / "protocol.json")
        write_json(self.run / "test_started.json", {"protocol_sha256": sha256_file(self.run / "protocol.json")})
        threshold = protocol["selected_threshold"]
        bright = [r for r in self.bright if r["split"] == "test"]
        records = stable_sample([r for r in self.exdark if r["split"] == "test"], len(bright), 2026) + bright
        selected_policy = "none" if threshold == 0 else "always" if threshold == 256 else "adaptive"
        methods = [("D_raw", "D", "none", 0), ("E_always", "E", "always", 256),
                   ("M_raw", "M", "none", 0), ("M_always", "M", "always", 256),
                   ("M_selected", "M", selected_policy, threshold)]
        rows = []
        for name, model_name, policy, thr in methods:
            print("속도 측정:", name)
            timing = benchmark_model(str(self.best(model_name)), records, self.pipeline(policy, thr),
                                     imgsz=self.imgsz, device=self.device, n=300, repeats=3,
                                     warmup=10, seed=2026)
            rows.append({"experiment": name, **timing_summary(timing)})
            write_json(self.results / "private" / f"{name}_timing.json", timing)
            write_csv(self.results / "latency.csv", rows)
        import pandas as pd
        return pd.DataFrame(rows)

    def report(self):
        from .report import make_report
        if self.mode != "full":
            print("작동 확인이 끝났습니다. MODE='full'로 바꿔 0단계부터 다시 실행하세요.")
            return None
        return make_report(self.run)

    def demo(self, image_path, model_name="M", threshold=None):
        import matplotlib.pyplot as plt
        from ultralytics import YOLO
        if threshold is None:
            threshold = read_json(self.run / "protocol.json")["selected_threshold"] if (self.run / "protocol.json").is_file() else 60
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        model = YOLO(str(self.best(model_name)))
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for axis, policy, label in zip(axes, ["none", "always", "adaptive"], ["Raw", "Always enhance", "Selective enhance"]):
            converted, used = self.pipeline(policy, threshold)(rgb)
            result = model.predict(converted[:, :, ::-1].copy(), imgsz=self.imgsz, device=self.device,
                                   conf=0.25, iou=0.7, verbose=False)[0]
            axis.imshow(result.plot()[:, :, ::-1])
            axis.set_title(f"{label} | enhanced={used}")
            axis.axis("off")
        plt.tight_layout()
        print("데모 confidence=0.25; 정량 평가 confidence=0.001. 박스 수만으로 성능을 판단하지 마세요.")
        return fig

    def export(self):
        if self.mode != "full" or not (self.run / "test_complete.json").exists():
            raise RuntimeError("본 실험의 최종 평가를 마친 뒤 제출 파일을 만드세요.")
        archive = self.run / "results_for_github.zip"
        allowed = []
        for name in ["config.json", "environment.json", "data_audit.json", "bright_audit.json", "protocol.json", "training_data.json"]:
            path = self.run / name
            if path.is_file():
                allowed.append((path, "results/" + name))
        for path in self.results.iterdir():
            if path.is_file() and path.suffix.lower() in {".csv", ".md", ".json"}:
                allowed.append((path, "results/" + path.name))
        figures = self.results / "figures"
        if figures.exists():
            allowed.extend((p, "results/figures/" + p.name) for p in figures.glob("*.png"))
        for name in ["D", "E", "M"]:
            for filename in ["results.csv", "args.yaml"]:
                path = self.run / "runs" / name / filename
                if path.is_file():
                    allowed.append((path, f"results/training/{name}/{filename}"))
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path, relative in allowed:
                output.write(path, relative)
        print("제출용 압축 파일:", archive)
        print("집계표·그래프·설정·학습 로그만 포함합니다. 원본/보정 이미지·라벨·가중치는 포함하지 않습니다.")
        return str(archive)
