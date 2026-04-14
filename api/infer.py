"""
api/infer.py — Inference wrapper for ESRGAN / Pix2Pix
Called by main.py's /enhance endpoint.
Replace the stub sections with real model loading logic.
"""

import io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _generate_heatmap(feature_map: torch.Tensor, target_size: tuple) -> bytes:
    """Convert intermediate feature activations → RGB heatmap PNG."""
    heat = feature_map.mean(dim=1, keepdim=True)          # [1,1,H,W]
    heat = F.interpolate(heat, size=target_size, mode="bilinear", align_corners=False)
    heat = heat.squeeze().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    colormap = np.zeros((*heat.shape, 3), dtype=np.uint8)
    colormap[..., 0] = (heat * 255).astype(np.uint8)       # red channel
    colormap[..., 1] = ((1 - heat) * 128).astype(np.uint8) # green tint
    img = Image.fromarray(colormap)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Model stubs (replace with real checkpoints) ───────────────────────────────

class _ESRGANStub(torch.nn.Module):
    """Placeholder — swap with real ESRGAN generator."""
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.conv(up)
        return up + 0.05 * feat, feat   # (output, intermediate features)


class _Pix2PixStub(torch.nn.Module):
    """Placeholder — swap with real Pix2Pix generator."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        feat = self.conv(x)
        return torch.tanh(x + 0.1 * feat), feat


# ── Main inference class ──────────────────────────────────────────────────────

class EnhancementInfer:
    def __init__(self, device: str = "auto"):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        ) if device == "auto" else torch.device(device)

        # Cache loaded models {key: model}
        self._models: dict = {}

    def _get_model(self, model: str, scale: int):
        key = f"{model}_{scale}"
        if key not in self._models:
            if model == "esrgan":
                m = _ESRGANStub(scale=scale)
                # TODO: m.load_state_dict(torch.load("checkpoints/esrgan_x4.pth"))
            elif model == "pix2pix":
                m = _Pix2PixStub()
                # TODO: m.load_state_dict(torch.load("checkpoints/pix2pix.pth"))
            else:
                raise ValueError(f"Unknown model: {model}")
            m.eval().to(self.device)
            self._models[key] = m
        return self._models[key]

    @torch.no_grad()
    def run(
        self,
        image_bytes: bytes,
        model: str = "esrgan",
        scale: int = 4,
        strength: float = 1.0,
    ) -> dict:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = img.size

        tensor = _pil_to_tensor(img).to(self.device)
        net = self._get_model(model, scale)
        enhanced_tensor, feat = net(tensor)

        # Blend with original for strength < 1.0
        if strength < 1.0:
            orig_up = F.interpolate(
                tensor, size=enhanced_tensor.shape[-2:],
                mode="bicubic", align_corners=False
            )
            enhanced_tensor = strength * enhanced_tensor + (1 - strength) * orig_up

        enhanced_bytes = _tensor_to_bytes(enhanced_tensor.cpu())
        heatmap_bytes = _generate_heatmap(feat.cpu(), (orig_h, orig_w))

        enh_img = Image.open(io.BytesIO(enhanced_bytes))
        return {
            "enhanced": enhanced_bytes,
            "heatmap": heatmap_bytes,
            "original_size": [orig_w, orig_h],
            "enhanced_size": list(enh_img.size),
        }
