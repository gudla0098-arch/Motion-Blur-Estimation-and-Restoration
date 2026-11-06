from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("MOTION_OUTPUT_DIR", "/tmp/motion-blur-restoration"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(
    title="Motion Blur Estimation and Restoration API",
    version="1.0.0",
    description="Model-based motion blur restoration using PSF priors and Wiener deconvolution.",
)


def _load_rgb_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload))
        if (image.format or "").upper() not in {"JPEG", "PNG", "WEBP"}:
            raise HTTPException(status_code=415, detail="Upload PNG, JPG or WEBP.")
        image.load()
        image = image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Image exceeds the 4 million pixel limit.")
    return image


def _motion_kernel(kind: str, length: int, angle: float) -> np.ndarray:
    """Build a normalized PSF for a selectable approximate motion model."""
    length = int(np.clip(length, 3, 45))
    radius = max(2, length // 2)
    side = 2 * radius + 1
    kernel = np.zeros((side, side), dtype=np.float64)
    center = radius

    if kind in {"linear", "uniform"}:
        theta = np.deg2rad(angle)
        xs = np.linspace(-radius, radius, length)
        weights = (
            np.exp(-0.5 * (xs / max(radius / 2, 1)) ** 2)
            if kind == "linear"
            else np.ones_like(xs)
        )
        for x, weight in zip(xs, weights):
            px = int(round(center + x * np.cos(theta)))
            py = int(round(center + x * np.sin(theta)))
            if 0 <= px < side and 0 <= py < side:
                kernel[py, px] += float(weight)
    elif kind == "circular":
        arc = np.deg2rad(min(length * 2.0, 100))
        for value in np.linspace(-arc, arc, length * 3):
            px = int(round(center + radius * np.cos(value)))
            py = int(round(center + radius * np.sin(value)))
            if 0 <= px < side and 0 <= py < side:
                kernel[py, px] += 1.0
    elif kind == "non-uniform":
        theta = np.deg2rad(angle)
        for value in np.linspace(-1.0, 1.0, length * 2):
            x = value * radius
            y = 0.32 * radius * np.sin(np.pi * value)
            px = int(round(center + x * np.cos(theta) - y * np.sin(theta)))
            py = int(round(center + x * np.sin(theta) + y * np.cos(theta)))
            if 0 <= px < side and 0 <= py < side:
                kernel[py, px] += 1.0
    else:
        raise HTTPException(
            status_code=422,
            detail="Blur type must be linear, circular, uniform or non-uniform.",
        )

    total = float(kernel.sum())
    if total <= 0:
        kernel[center, center] = 1.0
        total = 1.0
    return kernel / total


def _wiener_restore(image_array: np.ndarray, kernel: np.ndarray, strength: float) -> np.ndarray:
    """Restore RGB channels using stabilized frequency-domain Wiener filtering."""
    height, width, _ = image_array.shape
    if kernel.shape[0] > height or kernel.shape[1] > width:
        raise HTTPException(status_code=422, detail="Image is too small for the selected motion length.")
    padded_kernel = np.zeros((height, width), dtype=np.float64)
    kh, kw = kernel.shape
    padded_kernel[:kh, :kw] = kernel
    padded_kernel = np.roll(padded_kernel, -(kh // 2), axis=0)
    padded_kernel = np.roll(padded_kernel, -(kw // 2), axis=1)
    transfer = np.fft.fft2(padded_kernel)
    conjugate = np.conj(transfer)
    denominator = np.abs(transfer) ** 2 + max(0.0001, float(strength))
    restored = np.empty_like(image_array, dtype=np.float64)
    for channel in range(3):
        source = image_array[:, :, channel].astype(np.float64) / 255.0
        estimate = np.fft.ifft2((conjugate / denominator) * np.fft.fft2(source)).real
        restored[:, :, channel] = estimate * 255.0
    return np.clip(np.rint(restored), 0, 255).astype(np.uint8)


def _estimate_kernel_length(image_array: np.ndarray, prior_length: int) -> int:
    """Constrain the user-provided blur-length prior to valid image dimensions."""
    shortest_side = min(image_array.shape[:2])
    if shortest_side < 5:
        raise HTTPException(status_code=422, detail="Image must be at least 5 × 5 pixels.")
    return int(np.clip(prior_length, 3, min(45, shortest_side - 1)))


@app.get("/api/health")
def health_check() -> dict:
    return {"ok": True, "service": "Motion Blur Estimation and Restoration API"}


@app.post("/api/motion-blur")
async def motion_blur_restore(
    image: UploadFile = File(...),
    blur_type: Literal["linear", "circular", "uniform", "non-uniform"] = Form("linear"),
    length: int = Form(18),
    angle: float = Form(0),
    strength: float = Form(0.008),
) -> dict:
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Please choose an image first.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be smaller than 12 MB.")
    if not 0 <= angle <= 180:
        raise HTTPException(status_code=422, detail="Angle must be between 0 and 180 degrees.")
    if not 0.001 <= strength <= 0.05:
        raise HTTPException(status_code=422, detail="Restoration strength must be between 0.001 and 0.05.")

    original = _load_rgb_image(payload)
    original_array = np.asarray(original, dtype=np.uint8)
    estimated_length = _estimate_kernel_length(original_array, length)
    kernel = _motion_kernel(blur_type, estimated_length, angle)
    restored_array = _wiener_restore(original_array, kernel, strength)

    output_name = f"motion_restored_{uuid.uuid4().hex}.webp"
    output_path = OUTPUT_DIR / output_name
    Image.fromarray(restored_array, mode="RGB").save(output_path, format="WEBP", quality=94, method=5)
    return {
        "blur_type": blur_type,
        "motion_length_px": estimated_length,
        "angle_degrees": angle,
        "kernel_size": int(kernel.shape[0]),
        "method": "Wiener deconvolution (user-supplied PSF prior)",
        "processed_url": f"/api/result/{output_name}",
        "output_size_kb": round(output_path.stat().st_size / 1024, 2),
    }


@app.get("/api/result/{filename}")
def get_processed_image(filename: str) -> FileResponse:
    if Path(filename).name != filename or not filename.startswith("motion_restored_") or not filename.endswith(".webp"):
        raise HTTPException(status_code=404, detail="Result not found.")
    output_path = OUTPUT_DIR / filename
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="This result has expired or does not exist.")
    return FileResponse(output_path, media_type="image/webp")


@app.get("/")
@app.get("/motion-blur")
def motion_blur_page() -> FileResponse:
    return FileResponse(ROOT_DIR / "motion-blur.html")


@app.get("/motion-blur.css")
def motion_blur_stylesheet() -> FileResponse:
    return FileResponse(ROOT_DIR / "motion-blur.css", media_type="text/css")


@app.get("/motion-blur.js")
def motion_blur_javascript() -> FileResponse:
    return FileResponse(ROOT_DIR / "motion-blur.js", media_type="application/javascript")
