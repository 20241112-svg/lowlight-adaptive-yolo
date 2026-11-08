"""Small, offline tests for annotation geometry and split integrity."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from src.data import (
    COCO_CATEGORY_IDS, NAMES, DataPreparationError, brightness_rgb,
    parse_exdark_annotations, parse_exdark_metadata, prepare_coco_bright,
    prepare_exdark, safe_extract_zip,
)


class DataTests(unittest.TestCase):
    def test_luminance_uses_rgb_and_no_padding(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        self.assertEqual(brightness_rgb(image), 76.0)
        self.assertEqual(brightness_rgb(Image.new("RGB", (2, 2), "white")), 255.0)
        with self.assertRaises(ValueError):
            brightness_rgb(np.ones((2, 2, 3), dtype=np.float32))

    def test_multiobject_boxes_and_clipping(self):
        with tempfile.TemporaryDirectory() as root:
            label = Path(root) / "image.JPG.txt"
            label.write_text("% bbGt version=3\nPeople 10 20 30 40 0 0 0 0 0 0 0\n"
                             "Bicycle -2 5 12 10 0 0 0 0 0 0 0\n", encoding="utf-8")
            issues = []
            result = parse_exdark_annotations(label, 100, 80, issues)
            self.assertEqual(result, [[10, 10.0, 20.0, 40.0, 60.0], [0, 0.0, 5.0, 10.0, 15.0]])
            self.assertEqual(len(issues), 1)
            label.write_text("% bbGt version=3\nPeople 200 2 10 10\n")
            with self.assertRaises(DataPreparationError):
                parse_exdark_annotations(label, 100, 80)

    def test_metadata_no_duplicate_identity_across_splits(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "imageclasslist.txt"
            path.write_text("Name | Class | Light | In/Out | Train/Val/Test\n"
                            "One.JPG 1 2 1 1\nTwo.png 11 10 2 3\n")
            rows = parse_exdark_metadata(path)
            self.assertEqual(rows[0]["split"], "train")
            self.assertEqual(rows[1]["light"], "Twilight")
            path.write_text("One.jpg 1 2 1 1\none.PNG 1 2 1 3\n")
            with self.assertRaises(DataPreparationError):
                parse_exdark_metadata(path)

    def test_exdark_conversion_preserves_all_boxes_and_official_split(self):
        with tempfile.TemporaryDirectory() as root:
            raw, out = Path(root) / "raw", Path(root) / "out"
            (raw / "images" / "Nested").mkdir(parents=True)
            (raw / "labels").mkdir()
            (raw / "imageclasslist.txt").write_text(
                "Name | Class | Light | In/Out | Train/Val/Test\n"
                "one.JPG 1 2 1 1\ntwo.png 11 1 2 2\nthree.jpg 5 10 2 3\n")
            for name in ("one.jpg", "two.png", "three.jpg"):
                Image.new("RGB", (100, 80), (50, 50, 50)).save(raw / "images" / "Nested" / name)
                (raw / "labels" / (name + ".txt")).write_text(
                    "% bbGt version=3\nPeople 10 20 30 40 0 0 0 0 0 0 0\n"
                    "Car 50 10 20 10 0 0 0 0 0 0 0\n")
            records = prepare_exdark(raw, out, download=False, strict_counts=False)
            self.assertEqual([r["split"] for r in records], ["train", "val", "test"])
            self.assertEqual(len(records[0]["boxes"]), 2)
            rows = Path(records[0]["label_path"]).read_text().splitlines()
            self.assertEqual(rows[0], "10 0.2500000000 0.5000000000 0.3000000000 0.5000000000")
            self.assertTrue(all(Path(r["path"]).is_file() for r in records))
            cached = prepare_exdark(raw, out, download=False, strict_counts=False)
            self.assertEqual(records, cached)

    def test_missing_exdark_annotation_fails_not_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            raw = Path(root) / "raw"
            raw.mkdir()
            (raw / "imageclasslist.txt").write_text("one.jpg 1 1 1 1\n")
            Image.new("RGB", (10, 10)).save(raw / "one.jpg")
            with self.assertRaises(DataPreparationError):
                prepare_exdark(raw, Path(root) / "out", download=False, strict_counts=False)

    def test_coco_sparse_ids_all_objects_crowds_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as root:
            raw, out = Path(root) / "raw", Path(root) / "out"
            raw.mkdir()
            images, annotations = [], []
            for i in range(1, 7):
                name = f"{i:012d}.jpg"
                Image.new("RGB", (100, 80), (160, 160, 160)).save(raw / name)
                images.append({"id": i, "file_name": name, "width": 100, "height": 80})
                annotations.extend([
                    {"id": i * 10, "image_id": i, "category_id": 1, "bbox": [10, 20, 30, 40], "iscrowd": 0},
                    {"id": i * 10 + 1, "image_id": i, "category_id": 67, "bbox": [50, 10, 20, 10], "iscrowd": 0},
                    {"id": i * 10 + 2, "image_id": i, "category_id": 1, "bbox": [0, 0, 100, 80], "iscrowd": 1},
                    {"id": i * 10 + 3, "image_id": i, "category_id": 10, "bbox": [1, 1, 2, 2], "iscrowd": 0},
                ])
            data = {"images": images, "annotations": annotations,
                    "categories": [{"id": i, "name": name} for i, name in zip(COCO_CATEGORY_IDS, NAMES)]}
            (raw / "instances_val2017.json").write_text(json.dumps(data))
            records = prepare_coco_bright(raw, out, 2, 3, seed=9, download=False)
            val = {r["id"] for r in records if r["split"] == "val"}
            test = {r["id"] for r in records if r["split"] == "test"}
            self.assertEqual((len(val), len(test), len(val & test)), (2, 3, 0))
            self.assertEqual({box[0] for box in records[0]["boxes"]}, {10, 11})
            self.assertEqual(len(records[0]["boxes_crowd"]), 1)
            other = prepare_coco_bright(raw, Path(root) / "same_seed", 2, 3, seed=9, download=False)
            self.assertEqual([r["id"] for r in records], [r["id"] for r in other])
            (out / "keep_my_notes.txt").write_text("unrelated file")
            replaced = prepare_coco_bright(raw, out, 1, 1, seed=8, download=False, force=True)
            self.assertEqual(len(replaced), 2)
            self.assertEqual(len(list((out / "images").rglob("*.jpg"))), 2)
            self.assertEqual(len(list((out / "labels").rglob("*.txt"))), 2)
            self.assertTrue((out / "keep_my_notes.txt").is_file())
            with self.assertRaises(DataPreparationError):
                prepare_coco_bright(raw, Path(root) / "too_many", 4, 4, download=False)

    def test_zip_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../outside.txt", "bad")
            with self.assertRaises(DataPreparationError):
                safe_extract_zip(archive, Path(root) / "out")
            self.assertFalse((Path(root) / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
