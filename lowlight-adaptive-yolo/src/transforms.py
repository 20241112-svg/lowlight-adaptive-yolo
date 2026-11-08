"""Identical RGB preprocessing for offline training, online AP, and latency.

Zero-DCE++ model.py and weights are downloaded from the authors, not bundled.
All policies resize with the same PIL method before enhancement. Brightness is
computed on that unpadded image. Zero-DCE++ is replicate-padded, never cropped
geometrically, so bounding-box labels remain valid.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image

ZERODCE_REPO = "Li-Chongyi/Zero-DCE_extension"
ZERODCE_REV = "09f202b690f82da939b8e6ec8535960ae97ad8bd"


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size:
        return
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "lowlight-lab/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    temporary.replace(target)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def setup_zerodce(directory: str | Path) -> Path:
    directory = Path(directory).resolve()
    base = f"https://raw.githubusercontent.com/{ZERODCE_REPO}/{ZERODCE_REV}/Zero-DCE%2B%2B"
    _download(base + "/model.py", directory / "model.py")
    _download(base + "/snapshots_Zero_DCE%2B%2B/Epoch99.pth", directory / "Epoch99.pth")
    if (directory / "Epoch99.pth").stat().st_size < 1000:
        raise RuntimeError("Zero-DCE++ 가중치 다운로드를 확인하세요.")
    meta = {"repository": ZERODCE_REPO, "revision": ZERODCE_REV,
            "model_sha256": sha256_file(directory / "model.py"),
            "weights_sha256": sha256_file(directory / "Epoch99.pth")}
    (directory / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return directory / "Epoch99.pth"


def resize_rgb(rgb: np.ndarray, max_side: int = 640) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Input must be a nonempty uint8 HxWx3 RGB image.")
    h, w = rgb.shape[:2]
    if max_side > 0 and max(h, w) > max_side:
        ratio = max_side / max(h, w)
        size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
        rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))
    return np.ascontiguousarray(rgb)


def mean_brightness(rgb: np.ndarray) -> float:
    return float(np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32).mean())


class TransformPipeline:
    def __init__(self, policy="none", threshold=60.0, weights=None,
                 device="cuda", max_side=640, scale_factor=12,
                 gamma=0.5, model_path=None):
        if policy not in {"none", "gamma", "clahe", "always", "adaptive"}:
            raise ValueError(f"Unknown policy: {policy}")
        if not 0 <= float(threshold) <= 256:
            raise ValueError("threshold must be between 0 and 256")
        if int(scale_factor) < 1:
            raise ValueError("scale_factor must be >= 1")
        self.policy = policy
        self.threshold = float(threshold)
        self.weights = str(weights) if weights else None
        self.device = "cuda:0" if str(device) in {"0", "cuda"} else str(device)
        self.max_side = int(max_side)
        self.scale_factor = int(scale_factor)
        self.gamma = float(gamma)
        self._net = None
        self._clahe = None
        self.model_path = str(model_path) if model_path else None
        if policy in {"always", "adaptive"}:
            self._load_model()
        if policy == "clahe":
            import cv2
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _load_model(self):
        import torch
        if not self.weights or not Path(self.weights).is_file():
            raise FileNotFoundError("setup_zerodce()로 모델과 가중치를 먼저 준비하세요.")
        weight_path = Path(self.weights)
        candidates = [Path(self.model_path)] if self.model_path else [
            weight_path.parent / "model.py", weight_path.parent.parent / "model.py"]
        model_file = next((p for p in candidates if p.is_file()), None)
        if model_file is None:
            raise FileNotFoundError("가중치와 함께 공식 model.py 파일이 필요합니다.")
        spec = importlib.util.spec_from_file_location("_official_zero_dce_plus", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        net = module.enhance_net_nopool(self.scale_factor)
        state = torch.load(str(weight_path), map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=True)
        self._net = net.to(self.device).eval()
        for parameter in self._net.parameters():
            parameter.requires_grad_(False)

    def config(self) -> dict:
        return {"policy": self.policy, "threshold": self.threshold,
                "max_side": self.max_side, "scale_factor": self.scale_factor,
                "gamma": self.gamma, "device": self.device,
                "weights_sha256": sha256_file(self.weights) if self.weights else None,
                "brightness": "PIL L mean, unpadded, after shared resize",
                "resize": "PIL bilinear, keep aspect, never upscale"}

    def __call__(self, rgb: np.ndarray):
        rgb = resize_rgb(rgb, self.max_side)
        if self.policy == "none":
            return rgb, False
        if self.policy == "gamma":
            out = np.rint(255 * (rgb.astype(np.float32) / 255) ** self.gamma)
            return np.clip(out, 0, 255).astype(np.uint8), True
        if self.policy == "clahe":
            import cv2
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), True
        if self.policy == "adaptive" and mean_brightness(rgb) >= self.threshold:
            return rgb, False
        import torch
        import torch.nn.functional as functional
        h, w = rgb.shape[:2]
        with torch.inference_mode():
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(device=self.device, dtype=torch.float32) / 255.0
            pad_h, pad_w = (-h) % self.scale_factor, (-w) % self.scale_factor
            if pad_h or pad_w:
                tensor = functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
            enhanced, _ = self._net(tensor)
            enhanced = enhanced[:, :, :h, :w].clamp(0, 1)
            out = enhanced[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
        return np.ascontiguousarray(out), True
