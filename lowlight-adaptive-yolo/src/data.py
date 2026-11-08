"""Download and validate the original ExDark and COCO validation data.

No dataset is distributed with this project. Downloads happen only when the
user calls the preparation functions. All boxes use continuous ``xyxy`` pixel
coordinates; class IDs in the manifest and YOLO labels are always 0..11.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import stat
import time
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from PIL import Image
import yaml


NAMES = [
    "bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup",
    "dog", "motorcycle", "person", "dining table",
]
# These are Ultralytics' contiguous 80-class indices, NOT COCO JSON IDs.
EXDARK_TO_COCO = [1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]
# Official COCO instance-annotation category IDs have gaps.
COCO_CATEGORY_IDS = [2, 9, 44, 6, 3, 17, 62, 47, 18, 4, 1, 67]
EXDARK_CLASS_NAMES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup",
    "Dog", "Motorbike", "People", "Table",
]
LIGHTS = [
    "Low", "Ambient", "Object", "Single", "Weak", "Strong", "Screen",
    "Window", "Shadow", "Twilight",
]
EXDARK_META_URL = (
    "https://raw.githubusercontent.com/cs-chan/"
    "Exclusively-Dark-Image-Dataset/master/Groundtruth/imageclasslist.txt"
)
EXDARK_DOWNLOADS = {
    "ExDark.zip": "1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx",
    "ExDark_Annno.zip": "1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CONVERTER_VERSION = 1


class DataPreparationError(RuntimeError):
    """An actionable data problem; no images are silently omitted."""


def brightness_rgb(rgb) -> float:
    """Mean PIL BT.601 luminance, 0..255, measured before any padding.

    ``rgb`` must be a PIL image or an uint8 HxWx3 RGB NumPy array. Keeping this
    definition in one function ensures preparation and online routing agree.
    """
    if isinstance(rgb, Image.Image):
        image = rgb.convert("RGB")
    else:
        array = np.asarray(rgb)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("brightness_rgb expects a uint8 HxWx3 RGB image.")
        image = Image.fromarray(array, "RGB")
    return float(np.asarray(image.convert("L"), dtype=np.float64).mean())


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _download_http(url: str, destination: Path) -> Path:
    """Atomic download; an interrupted file is never accepted as a cache."""
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "lowlight-adaptive-yolo/1.0"})
            with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                expected = response.headers.get("Content-Length")
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            size = temporary.stat().st_size
            if not size or (expected and size != int(expected)):
                raise DataPreparationError(f"Incomplete download: {destination.name}")
            temporary.replace(destination)
            return destination
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise DataPreparationError(
                    f"Download failed: {destination.name}. Open {url} in a browser, "
                    f"save the file as {destination}, and run this cell again. ({exc})"
                ) from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _download_gdrive(file_id: str, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size:
        return destination
    try:
        import gdown
    except ImportError as exc:
        raise DataPreparationError("Run the notebook's installation cell first (gdown is missing).") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        result = gdown.download(id=file_id, output=str(temporary), quiet=False)
        if result is None or not temporary.is_file() or not zipfile.is_zipfile(temporary):
            raise DataPreparationError("The response was not a ZIP archive.")
        temporary.replace(destination)
        return destination
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise DataPreparationError(
            "Google Drive download failed or reached its quota. Download "
            f"https://drive.google.com/file/d/{file_id}/view manually and save it "
            f"as {destination}. Then rerun this cell. ({exc})"
        ) from exc


def safe_extract_zip(archive, destination) -> Path:
    """Extract a ZIP safely and cache only a completed extraction.

    A stale or interrupted extraction is rebuilt in a sibling temporary
    directory. File names cannot escape the destination and ZIP symlinks are
    rejected. Nested ZIPs are handled by ``_extract_nested``.
    """
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    fingerprint = {"size": archive.stat().st_size, "mtime_ns": archive.stat().st_mtime_ns}
    marker = destination / ".extraction_complete.json"
    if marker.is_file():
        try:
            if json.loads(marker.read_text()) == fingerprint:
                return destination
        except (OSError, ValueError):
            pass
    if not zipfile.is_zipfile(archive):
        raise DataPreparationError(f"Not a ZIP file: {archive}. Download it again.")
    staging = destination.with_name(destination.name + ".extracting")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            if sum(info.file_size for info in bundle.infolist()) > 30 * 1024**3:
                raise DataPreparationError(f"Unexpectedly large ZIP: {archive}")
            for info in bundle.infolist():
                name = info.filename.replace("\\", "/")
                pieces = PurePosixPath(name)
                if pieces.is_absolute() or ".." in pieces.parts or any(":" in p for p in pieces.parts[:1]):
                    raise DataPreparationError(f"Unsafe path in ZIP: {info.filename}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise DataPreparationError(f"ZIP contains a symbolic link: {info.filename}")
                target = staging.joinpath(*pieces.parts)
                if not target.resolve().is_relative_to(staging.resolve()):
                    raise DataPreparationError(f"Unsafe path in ZIP: {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
        _write_json(staging / marker.name, fingerprint)
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _extract_nested(archive: Path, output: Path, depth: int = 0) -> None:
    folder = safe_extract_zip(archive, output)
    nested = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".zip"]
    if depth >= 3 and nested:
        raise DataPreparationError(f"Too many nested ZIP levels under {archive}.")
    for child in nested:
        if ".nested" in child.relative_to(folder).parts:
            continue
        key = hashlib.sha256(str(child.relative_to(folder)).encode()).hexdigest()[:12]
        _extract_nested(child, folder / ".nested" / key, depth + 1)


def _file_index(root: Path, suffixes: set[str]) -> dict[str, list[Path]]:
    found = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes and "__MACOSX" not in path.parts:
            found[path.name.casefold()].append(path)
    return dict(found)


def _unique_match(index: dict[str, list[Path]], names: list[str], description: str) -> Path:
    matches = sorted({p for name in names for p in index.get(name.casefold(), [])})
    if not matches:
        raise DataPreparationError(f"Missing {description}; tried {names}.")
    if len(matches) > 1:
        checksums = {_sha256(p) for p in matches}
        if len(checksums) != 1:
            raise DataPreparationError(f"Conflicting duplicate files for {description}: {matches}")
    return matches[0]


def parse_exdark_metadata(path) -> list[dict]:
    records = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(("#", "%")) or line.casefold().startswith("name"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise DataPreparationError(f"Metadata line {number}: expected 5 columns, got {line!r}.")
        filename = parts[0]
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise DataPreparationError(f"Unsafe image name in metadata: {filename}")
        try:
            cls, light, scene, split = map(int, parts[1:])
        except ValueError as exc:
            raise DataPreparationError(f"Invalid metadata at line {number}: {line!r}") from exc
        if not (1 <= cls <= 12 and 1 <= light <= 10 and scene in (1, 2) and split in (1, 2, 3)):
            raise DataPreparationError(f"Out-of-range metadata at line {number}: {line!r}")
        identity = Path(filename).stem.casefold()
        if identity in seen:
            raise DataPreparationError(f"Duplicate image identity in metadata: {filename}")
        seen.add(identity)
        records.append({"filename": filename, "primary_class": cls - 1, "light": LIGHTS[light - 1],
                        "scene": "indoor" if scene == 1 else "outdoor",
                        "split": {1: "train", 2: "val", 3: "test"}[split]})
    if not records:
        raise DataPreparationError(f"No image records in {path}.")
    return records


def _clip_xywh(cls: int, xywh, width: int, height: int, context: str, issues: list) -> list:
    x, y, w, h = map(float, xywh)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise DataPreparationError(f"Invalid box at {context}: {xywh}")
    original = [x, y, x + w, y + h]
    clipped = [max(0.0, min(float(width), original[0])), max(0.0, min(float(height), original[1])),
               max(0.0, min(float(width), original[2])), max(0.0, min(float(height), original[3]))]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise DataPreparationError(f"Box has zero area after clipping at {context}: {xywh}")
    if original != clipped:
        issues.append({"kind": "box_clipped", "context": context, "original_xyxy": original,
                       "clipped_xyxy": clipped})
    return [cls] + clipped


def parse_exdark_annotations(path, width: int, height: int, issues=None) -> list[list]:
    """Use the source ``[left, top, width, height]`` without guessing an offset.

    The official README does not prescribe an extra +1 or -1 conversion.
    Consequently x2=left+width and y2=top+height; boxes are clipped to image
    bounds and every clipping is recorded for the notebook's visual inspection.
    """
    issues = issues if issues is not None else []
    mapping = {name.casefold(): i for i, name in enumerate(EXDARK_CLASS_NAMES)}
    mapping.update({"person": 10, "motorcycle": 9, "diningtable": 11})
    boxes = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(("%", "#")):
            continue
        parts = line.split()
        if len(parts) < 5 or parts[0].casefold() not in mapping:
            raise DataPreparationError(f"Unknown/malformed annotation at {path}:{number}: {line!r}")
        try:
            box = _clip_xywh(mapping[parts[0].casefold()], parts[1:5], width, height,
                             f"{Path(path).name}:{number}", issues)
        except (ValueError, TypeError) as exc:
            raise DataPreparationError(f"Non-numeric box at {path}:{number}") from exc
        boxes.append(box)
    if not boxes:
        raise DataPreparationError(f"No valid object boxes in {path}.")
    return boxes


def _write_yolo(path: Path, boxes, width: int, height: int) -> None:
    rows = []
    for cls, x1, y1, x2, y2 in boxes:
        xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        rows.append(f"{int(cls)} {xc:.10f} {yc:.10f} {bw:.10f} {bh:.10f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _copy_image(source: Path, out_dir: Path, identity: str, issues: list):
    """Keep original bytes, normalizing unusual EXIF orientation explicitly."""
    try:
        with Image.open(source) as original:
            original.load()
            rgb = original.convert("RGB")
            width, height = rgb.size
            brightness = brightness_rgb(rgb)
            orientation = original.getexif().get(274, 1)
            # Ground truth is in the decoded source coordinate system. Strip
            # orientation metadata rather than letting a reader rotate pixels.
            normalize = orientation not in (None, 1) or original.mode == "CMYK"
            suffix = ".png" if normalize else source.suffix.lower()
            destination = out_dir / (identity + suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if normalize:
                rgb.info.clear()
                rgb.save(destination, format="PNG")
                issues.append({"kind": "image_metadata_normalized", "image": source.name,
                               "exif_orientation": orientation, "source_mode": original.mode,
                               "note": "Pixels were not rotated; source box coordinates preserved."})
            elif source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        return destination, width, height, brightness
    except (OSError, ValueError) as exc:
        raise DataPreparationError(f"Cannot read image {source}: {exc}") from exc


def _save_dataset(out: Path, records: list[dict], settings: dict, report: dict) -> None:
    splits = {record["split"] for record in records}
    config = {"path": str(out), "names": dict(enumerate(NAMES))}
    for split in ("train", "val", "test"):
        if split in splits:
            config[split] = f"images/{split}"
    (out / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                                      encoding="utf-8")
    _write_json(out / "manifest.json", records)
    _write_json(out / "preparation_report.json", {"settings": settings, **report})


def _clear_generated_dataset(out: Path) -> None:
    """Clear only this converter's outputs after source validation succeeds.

    In particular, a changed COCO sample must not leave old images in a split
    directory where a directory-based YOLO loader could discover them.
    """
    for name in ("images", "labels"):
        target = out / name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    for name in ("manifest.json", "dataset.yaml", "preparation_report.json"):
        (out / name).unlink(missing_ok=True)


def _cached_records(out: Path, settings: dict, force: bool):
    if force or not (out / "manifest.json").is_file() or not (out / "preparation_report.json").is_file():
        return None
    report = json.loads((out / "preparation_report.json").read_text(encoding="utf-8"))
    if report.get("settings") != settings:
        raise DataPreparationError(
            f"{out} was prepared with different settings. Use a new output folder "
            "to preserve the previous experiment, or pass force=True deliberately."
        )
    records = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if not records or len({r["id"] for r in records}) != len(records):
        raise DataPreparationError(f"Invalid cached manifest: {out / 'manifest.json'}")
    for record in records:
        # Backups can be restored under a different absolute project path.
        record["path"] = str(out / "images" / record["split"] / Path(record["path"]).name)
        record["label_path"] = str(out / "labels" / record["split"] / Path(record["label_path"]).name)
        if not Path(record["path"]).is_file() or not Path(record["label_path"]).is_file():
            raise DataPreparationError(f"Incomplete cached dataset at {out}. Rebuild with force=True.")
    _save_dataset(out, records, settings, {k: v for k, v in report.items() if k != "settings"})
    print(f"Reusing {len(records):,} validated images from {out}.")
    return records


def prepare_exdark(raw_dir, output_dir, download: bool = True, *, strict_counts: bool = True,
                   force: bool = False) -> list[dict]:
    """Prepare original ExDark with its official split and all object boxes.

    Manual fallback: place ``ExDark.zip``, ``ExDark_Annno.zip`` and, if needed,
    ``imageclasslist.txt`` in ``raw_dir``. Extracted files under ``raw_dir`` are
    also accepted. ``strict_counts=False`` exists for small synthetic tests,
    never for silently accepting an incomplete real download.
    """
    raw, out = Path(raw_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    settings = {"dataset": "exdark", "converter_version": CONVERTER_VERSION,
                "strict_counts": strict_counts, "coordinates": "source_l_t_w_h_no_offset"}
    cached = _cached_records(out, settings, force)
    if cached is not None:
        return cached
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    metadata_candidates = list(raw.rglob("imageclasslist.txt"))
    if not metadata_candidates:
        if not download:
            raise DataPreparationError(f"Missing {raw / 'imageclasslist.txt'}")
        _download_http(EXDARK_META_URL, raw / "imageclasslist.txt")
        metadata_candidates = [raw / "imageclasslist.txt"]
    metadata_path = _unique_match({"imageclasslist.txt": metadata_candidates}, ["imageclasslist.txt"],
                                  "official ExDark split metadata")
    metadata = parse_exdark_metadata(metadata_path)
    counts = dict(Counter(row["split"] for row in metadata))
    expected = {"train": 3000, "val": 1800, "test": 2563}
    if strict_counts and counts != expected:
        raise DataPreparationError(f"Unexpected official split counts: {counts}; expected {expected}.")

    # Use extracted local data first. If incomplete, extract existing archives
    # or obtain the official archives. Never substitute a random third-party split.
    image_index = _file_index(raw, IMAGE_SUFFIXES)
    label_index = _file_index(raw, {".txt"})
    images_missing = any(row["filename"].casefold() not in image_index for row in metadata)
    labels_missing = any(not any(name.casefold() in label_index for name in
                                [row["filename"] + ".txt", Path(row["filename"]).stem + ".txt"])
                         for row in metadata)
    for filename, needed in (("ExDark.zip", images_missing), ("ExDark_Annno.zip", labels_missing)):
        if not needed:
            continue
        archive = raw / filename
        if not archive.is_file():
            if not download:
                raise DataPreparationError(f"Missing extracted files and {archive}.")
            _download_gdrive(EXDARK_DOWNLOADS[filename], archive)
        _extract_nested(archive, raw / "extracted" / Path(filename).stem)
    image_index, label_index = _file_index(raw, IMAGE_SUFFIXES), _file_index(raw, {".txt"})
    resolved, errors = [], []
    for row in metadata:
        try:
            image = _unique_match(image_index, [row["filename"]], f"image {row['filename']}")
            label = _unique_match(label_index, [row["filename"] + ".txt", Path(row["filename"]).stem + ".txt"],
                                  f"annotation for {row['filename']}")
            resolved.append((row, image, label))
        except DataPreparationError as exc:
            errors.append(str(exc))
    if errors:
        _write_json(out / "preparation_errors.json", errors)
        raise DataPreparationError(
            f"{len(errors)} images/annotations are missing or ambiguous. No image was silently dropped. "
            f"See {out / 'preparation_errors.json'}. First error: {errors[0]}"
        )
    if force:
        _clear_generated_dataset(out)
    records, issues = [], []
    for index, (row, image, label) in enumerate(resolved, 1):
        identity = "exdark_" + Path(row["filename"]).stem
        target, width, height, brightness = _copy_image(image, out / "images" / row["split"], identity, issues)
        boxes = parse_exdark_annotations(label, width, height, issues)
        label_path = out / "labels" / row["split"] / (identity + ".txt")
        _write_yolo(label_path, boxes, width, height)
        records.append({"id": identity, "path": str(target), "label_path": str(label_path),
                        "split": row["split"], "source": "exdark", "group": "lowlight",
                        "light": row["light"], "scene": row["scene"],
                        "primary_class": row["primary_class"], "brightness": brightness,
                        "width": width, "height": height, "boxes": boxes, "boxes_crowd": []})
        if index % 500 == 0:
            print(f"ExDark conversion: {index:,}/{len(resolved):,}")
    report = {"counts": counts, "images": len(records), "boxes": sum(len(r["boxes"]) for r in records),
              "metadata_sha256": _sha256(metadata_path), "issues": issues,
              "label_policy": "Keep every annotated object. Clip out-of-bounds boxes and record each clipping.",
              "source": "https://github.com/cs-chan/Exclusively-Dark-Image-Dataset"}
    _save_dataset(out, records, settings, report)
    print(f"ExDark prepared: {counts}. Recorded clipping/normalization issues: {len(issues)}.")
    return records


def prepare_coco_bright(raw_dir, output_dir, n_val: int = 300, n_test: int = 500,
                        brightness_min: float = 90, seed: int = 2026, *, download: bool = True,
                        force: bool = False) -> list[dict]:
    """Select disjoint bright validation/test subsets from COCO val2017.

    Bright is a luminance criterion, not a claim about time of day. All regular
    and crowd boxes of the 12 target classes are retained. Crowd boxes are in
    ``boxes_crowd`` and must be passed to the evaluator as ``iscrowd=1``. These
    images are evaluation-only and are not used to train the YOLO detector.
    """
    if not isinstance(n_val, int) or not isinstance(n_test, int) or min(n_val, n_test) < 1:
        raise ValueError("n_val and n_test must be positive integers.")
    if not 0 <= brightness_min <= 255:
        raise ValueError("brightness_min must be between 0 and 255.")
    raw, out = Path(raw_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    settings = {"dataset": "coco_val2017_bright", "converter_version": CONVERTER_VERSION,
                "n_val": n_val, "n_test": n_test, "brightness_min": float(brightness_min), "seed": seed}
    cached = _cached_records(out, settings, force)
    if cached is not None:
        return cached
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    annotation_candidates = list(raw.rglob("instances_val2017.json"))
    if not annotation_candidates:
        archive = raw / "annotations_trainval2017.zip"
        if not archive.is_file():
            if not download:
                raise DataPreparationError(f"Missing COCO instances_val2017.json under {raw}.")
            # COCO's official download page publishes HTTP links; its HTTPS
            # virtual host is not consistently available. These are public,
            # unauthenticated dataset files (no credentials are transmitted).
            _download_http("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
        safe_extract_zip(archive, raw / "extracted" / "annotations")
        annotation_candidates = list(raw.rglob("instances_val2017.json"))
    annotation_path = _unique_match({"instances_val2017.json": annotation_candidates},
                                   ["instances_val2017.json"], "COCO val2017 annotations")
    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in dataset["categories"]}
    for coco_id, name in zip(COCO_CATEGORY_IDS, NAMES):
        if categories.get(coco_id) != name:
            raise DataPreparationError(f"COCO category mismatch for ID {coco_id}: expected {name!r}.")
    class_map = {coco_id: index for index, coco_id in enumerate(COCO_CATEGORY_IDS)}
    by_image = defaultdict(list)
    for annotation in dataset["annotations"]:
        if int(annotation["category_id"]) in class_map:
            by_image[int(annotation["image_id"])].append(annotation)
    image_info = {int(item["id"]): item for item in dataset["images"]}
    if len(image_info) != len(dataset["images"]):
        raise DataPreparationError("Duplicate COCO image IDs in annotations.")
    image_index = _file_index(raw, IMAGE_SUFFIXES)
    if any(item["file_name"].casefold() not in image_index for item in image_info.values()):
        archive = raw / "val2017.zip"
        if not archive.is_file():
            if not download:
                raise DataPreparationError(f"Some COCO images are missing under {raw}.")
            _download_http("http://images.cocodataset.org/zips/val2017.zip", archive)
        safe_extract_zip(archive, raw / "extracted" / "images")
        image_index = _file_index(raw, IMAGE_SUFFIXES)
    candidates = sorted(image_info)
    random.Random(seed).shuffle(candidates)
    selected, issues = [], []
    for image_id in candidates:
        annotations = sorted(by_image[image_id], key=lambda a: a["id"])
        if not any(not ann.get("iscrowd", 0) for ann in annotations):
            continue
        info = image_info[image_id]
        source = _unique_match(image_index, [info["file_name"]], f"COCO image {image_id}")
        with Image.open(source) as image:
            image.load()
            if image.size != (int(info["width"]), int(info["height"])):
                raise DataPreparationError(f"Image/annotation dimension mismatch: {source}")
            brightness = brightness_rgb(image)
        if brightness < brightness_min:
            continue
        boxes, crowd_boxes = [], []
        for annotation in annotations:
            box = _clip_xywh(class_map[int(annotation["category_id"])], annotation["bbox"],
                             int(info["width"]), int(info["height"]),
                             f"COCO image {image_id}, annotation {annotation['id']}", issues)
            (crowd_boxes if annotation.get("iscrowd", 0) else boxes).append(box)
        selected.append((source, info, boxes, crowd_boxes))
        if len(selected) == n_val + n_test:
            break
    if len(selected) < n_val + n_test:
        raise DataPreparationError(
            f"Only {len(selected)} eligible bright images for {n_val + n_test} requested. "
            "Reduce n_val/n_test before choosing thresholds or running the final test. "
            "No images have been duplicated."
        )
    if force:
        _clear_generated_dataset(out)
    records = []
    for index, (source, info, boxes, crowd_boxes) in enumerate(selected):
        split = "val" if index < n_val else "test"
        identity = f"coco_{int(info['id']):012d}"
        target, width, height, brightness = _copy_image(source, out / "images" / split, identity, issues)
        label_path = out / "labels" / split / (identity + ".txt")
        # YOLO text format cannot represent crowd regions; the manifest is the
        # authoritative evaluation input and retains them separately.
        _write_yolo(label_path, boxes, width, height)
        records.append({"id": identity, "path": str(target), "label_path": str(label_path),
                        "split": split, "source": "coco_val2017", "group": "bright", "light": "Bright",
                        "brightness": brightness, "width": width, "height": height,
                        "boxes": boxes, "boxes_crowd": crowd_boxes})
    report = {"counts": {"val": n_val, "test": n_test}, "images": len(records),
              "source_split": "COCO val2017 only", "annotation_sha256": _sha256(annotation_path),
              "coco_image_ids_val": [int(r["id"].split("_")[-1]) for r in records if r["split"] == "val"],
              "coco_image_ids_test": [int(r["id"].split("_")[-1]) for r in records if r["split"] == "test"],
              "selection": "Seeded random image order; mean PIL luminance >= threshold and >=1 regular target box.",
              "crowd_policy": "All target crowd boxes retained in manifest boxes_crowd; evaluate with COCOeval.",
              "issues": issues, "source": "https://cocodataset.org/#download"}
    _save_dataset(out, records, settings, report)
    print(f"Bright COCO prepared: val={n_val}, test={n_test}; image IDs are disjoint.")
    return records
