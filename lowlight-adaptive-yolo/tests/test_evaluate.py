"""Small synthetic checks for evaluation mistakes that would invalidate results.

These fixtures are software QA, not experimental results. They require only
NumPy and Pillow; the official COCO metric test additionally uses pycocotools.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluate import (
    NAMES, EXDARK_TO_COCO, benchmark_model, build_coco_ground_truth,
    class_mapping, collect_predictions, evaluate_predictions,
    records_signature, select_threshold, stratified_sample,
)


def record(image_id, group="lowlight"):
    return {
        "id": image_id, "path": "unused", "split": "val", "source": "exdark",
        "group": group, "light": "Low", "width": 200, "height": 100,
        "boxes": [[10, 20, 10, 100, 80]],
    }


class FakeBoxes:
    def __init__(self):
        # Coordinates refer to the resized 100x50 input, not the 200x100 original.
        self.xyxy = np.array([[10.0, 5.0, 50.0, 40.0]])
        self.cls = np.array([10.0])
        self.conf = np.array([0.9])

    def __len__(self):
        return 1


class FakeResult:
    boxes = FakeBoxes()


class FakeDetector:
    names = dict(enumerate(NAMES))

    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult() for _ in kwargs["source"]]


class HalfSizePipeline:
    policy = "none"
    threshold = 0
    max_side = 100

    def __call__(self, rgb):
        return rgb[::2, ::2].copy(), False


class EvaluationProtocolTests(unittest.TestCase):
    def test_coco_ids_and_exdark_aliases_are_not_confused(self):
        coco_names = {i: f"unused_{i}" for i in range(80)}
        for target, coco_id in enumerate(EXDARK_TO_COCO):
            coco_names[coco_id] = NAMES[target]
        mapping = class_mapping(coco_names)
        self.assertEqual(mapping[0], 10)  # COCO person=0, project person=10.
        self.assertEqual(mapping[2], 4)   # COCO car=2, project car=4.
        aliases = list(NAMES)
        aliases[9:12] = ["Motorbike", "People", "Table"]
        self.assertEqual(class_mapping(aliases), dict(enumerate(range(12))))
        with self.assertRaises(ValueError):
            class_mapping({0: "person"})

    def test_rgb_bgr_and_resized_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "red.png"
            array = np.zeros((100, 200, 3), dtype=np.uint8)
            array[..., 0] = 255
            Image.fromarray(array).save(path)
            rec = record("red")
            rec["path"] = str(path)
            model = FakeDetector()
            result = collect_predictions(model, [rec], HalfSizePipeline(), device="cpu", progress=False)
            self.assertEqual(model.calls[0]["source"][0][0, 0].tolist(), [0, 0, 255])
            self.assertEqual(result[0]["bbox"], [20.0, 10.0, 80.0, 70.0])
            self.assertEqual(result[0]["category_id"], 11)
            self.assertEqual(result[0]["image_id"], 1)
            self.assertEqual(model.calls[0]["conf"], 0.001)

    def test_ground_truth_keeps_regular_and_crowd_annotations(self):
        rec = record("first")
        rec["boxes_crowd"] = [[10, 0, 0, 200, 100]]
        gt = build_coco_ground_truth([rec])
        self.assertEqual(gt["annotations"][0]["bbox"], [20.0, 10.0, 80.0, 70.0])
        self.assertEqual(gt["annotations"][0]["iscrowd"], 0)
        self.assertEqual(gt["annotations"][1]["iscrowd"], 1)
        with self.assertRaises(ValueError):
            build_coco_ground_truth([rec, rec])

    def test_order_sensitive_signature_and_sampling_without_replacement(self):
        rows = [record(f"low_{i}") for i in range(8)] + [record(f"bright_{i}", "bright") for i in range(4)]
        chosen = stratified_sample(rows, n=6, seed=7)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(len({r["id"] for r in chosen}), 6)
        self.assertEqual(sum(r["group"] == "bright" for r in chosen), 2)
        self.assertEqual(chosen, stratified_sample(rows, n=6, seed=7))
        self.assertNotEqual(records_signature(rows), records_signature(list(reversed(rows))))

    def test_threshold_selection_protects_each_domain_and_uses_latency(self):
        # Deliberately invented toy values to exercise the selection logic.
        specs = {
            0: (0.75, 0.70, 0.80, 5.0),
            40: (0.795, 0.755, 0.845, 6.0),  # overall OK; lowlight violates.
            80: (0.795, 0.795, 0.795, 7.0),
            120: (0.80, 0.80, 0.80, 8.0),
            256: (0.80, 0.80, 0.80, 10.0),
        }
        rows = []
        for threshold, (all_ap, low_ap, bright_ap, latency) in specs.items():
            for group, score in zip(("all", "lowlight", "bright"), (all_ap, low_ap, bright_ap)):
                rows.append({"threshold": threshold, "group": group, "mAP50": score, "mean_ms": latency, "split": "val"})
        result = select_threshold(rows, allowed_drop=0.01)
        self.assertEqual(result["threshold"], 80)
        rows[0]["split"] = "test"
        with self.assertRaises(ValueError):
            select_threshold(rows)

    def test_no_enhancement_can_win_and_mixed_manifests_are_rejected(self):
        rows = [
            {"threshold": 0, "group": "all", "mAP50": 0.8, "mean_ms": 5, "records_sha256": "same"},
            {"threshold": 256, "group": "all", "mAP50": 0.8, "mean_ms": 10, "records_sha256": "same"},
        ]
        self.assertEqual(select_threshold(rows)["threshold"], 0)
        rows[0]["records_sha256"] = "different"
        with self.assertRaises(ValueError):
            select_threshold(rows)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools not installed")
    def test_perfect_empty_and_pooled_coco_metrics(self):
        rows = [record("dark"), record("bright", "bright")]
        perfect = [
            {"image_id": i, "category_id": 11, "bbox": [20, 10, 80, 70], "score": 0.9}
            for i in (1, 2)
        ]
        results = evaluate_predictions(rows, perfect, include_lights=False)
        self.assertAlmostEqual(results[0]["mAP50"], 1.0)
        self.assertAlmostEqual(results[0]["mAP50_95"], 1.0)
        self.assertAlmostEqual(results[0]["AR100"], 1.0)
        empty = evaluate_predictions(rows, [], include_lights=False)
        self.assertAlmostEqual(empty[0]["mAP50"], 0.0)
        self.assertEqual({r["group"] for r in results}, {"all", "lowlight", "bright"})


if __name__ == "__main__":
    unittest.main()
