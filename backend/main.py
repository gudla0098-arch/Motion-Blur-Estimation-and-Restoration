from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import pywt
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn, idctn
from scipy.sparse.linalg import svds

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
OUTPUT_DIR = Path(os.environ.get("DIP_OUTPUT_DIR", "/tmp/dip-image-compression"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(
    title="DIP Image Compression API",
    version="1.0.0",
    description="Image compression using DCT, DWT and SVD.",
)


def _load_rgb_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload))
        image_format = (image.format or "").upper()
        if image_format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image format. Please upload PNG, JPG or WEBP.",
            )
        image.load()
        image = image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum supported size is {MAX_IMAGE_PIXELS:,} pixels.",
        )
    return image


def _dct_compress(image_array: np.ndarray, quality: int) -> np.ndarray:
    # JPEG-style 8x8 block DCT and coefficient quantization.
    luminance_table = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=np.float64,
    )
    quality = int(np.clip(quality, 10, 90))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    quant_table = np.maximum(1, np.floor((luminance_table * scale + 50) / 100))

    height, width, channels = image_array.shape
    pad_h = (-height) % 8
    pad_w = (-width) % 8
    reconstructed = np.empty((height, width, channels), dtype=np.float64)

    for channel_index in range(channels):
        channel = image_array[:, :, channel_index].astype(np.float64) - 128.0
        padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
        block_h, block_w = padded.shape[0] // 8, padded.shape[1] // 8
        blocks = padded.reshape(block_h, 8, block_w, 8).transpose(0, 2, 1, 3)

        coefficients = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
        quantized = np.round(coefficients / quant_table) * quant_table
        restored = idctn(quantized, type=2, norm="ortho", axes=(-2, -1))
        restored = restored.transpose(0, 2, 1, 3).reshape(padded.shape)
        reconstructed[:, :, channel_index] = restored[:height, :width] + 128.0

    return np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)


def _dwt_compress(
    image_array: np.ndarray,
    wavelet: str,
    threshold: float,
) -> np.ndarray:
    if wavelet not in {"haar", "db2", "sym2"}:
        raise HTTPException(status_code=422, detail="Wavelet must be haar, db2 or sym2.")

    result = np.empty_like(image_array, dtype=np.uint8)
    for channel_index in range(image_array.shape[2]):
        channel = image_array[:, :, channel_index].astype(np.float64)
        coefficients = pywt.wavedec2(channel, wavelet=wavelet, level=1)
        approximation, details = coefficients[0], coefficients[1]
        thresholded_details = tuple(
            pywt.threshold(detail, value=float(threshold), mode="soft")
            for detail in details
        )
        restored = pywt.waverec2(
            [approximation, thresholded_details],
            wavelet=wavelet,
        )
        restored = restored[: channel.shape[0], : channel.shape[1]]
        result[:, :, channel_index] = np.clip(np.rint(restored), 0, 255).astype(np.uint8)

    return result


def _svd_compress(image_array: np.ndarray, rank: int) -> tuple[np.ndarray, int, int]:
    height, width, channels = image_array.shape
    maximum_rank = min(height, width)
    if maximum_rank < 2:
        raise HTTPException(status_code=422, detail="Image dimensions are too small for SVD.")

    retained_rank = int(np.clip(rank, 1, maximum_rank))
    result = np.empty_like(image_array, dtype=np.uint8)

    for channel_index in range(channels):
        matrix = image_array[:, :, channel_index].astype(np.float64)
        try:
            if retained_rank < maximum_rank:
                u, singular_values, vt = svds(
                    matrix,
                    k=retained_rank,
                    which="LM",
                    return_singular_vectors=True,
                    rng=0,
                )
                order = np.argsort(singular_values)[::-1]
                u = u[:, order]
                singular_values = singular_values[order]
                vt = vt[order, :]
                restored = (u * singular_values) @ vt
            else:
                u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
                restored = (u[:, :retained_rank] * singular_values[:retained_rank]) @ vt[:retained_rank, :]
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"SVD could not process this image at the selected rank. Try a smaller rank. ({type(exc).__name__})",
            ) from exc

        result[:, :, channel_index] = np.clip(np.rint(restored), 0, 255).astype(np.uint8)

    return result, retained_rank, maximum_rank


def _make_metrics(
    original: np.ndarray,
    processed: np.ndarray,
    original_size: int,
    output_size: int,
) -> dict:
    difference = original.astype(np.float64) - processed.astype(np.float64)
    mse = float(np.mean(np.square(difference)))
    psnr: float | str
    if mse == 0:
        psnr = "Infinity"
    else:
        psnr = float(10 * np.log10((255.0 ** 2) / mse))

    bytes_saved = int(original_size - output_size)
    ratio = float(original_size / output_size) if output_size else 0.0
    reduction = float((bytes_saved / original_size) * 100) if original_size else 0.0

    return {
        "original_size_kb": round(original_size / 1024, 2),
        "processed_size_kb": round(output_size / 1024, 2),
        "compression_ratio": round(ratio, 4),
        "size_reduction_percent": round(reduction, 2),
        "bytes_saved": bytes_saved,
        "pixel_mse": round(mse, 6),
        "psnr_db": round(psnr, 4) if isinstance(psnr, float) else psnr,
    }


@app.get("/api/health")
def health_check() -> dict:
    return {"ok": True, "service": "DIP Image Compression API"}


@app.post("/api/process/{technique}")
async def process_image(
    technique: Literal["dct", "dwt", "svd"],
    image: UploadFile = File(...),
    quality: int = Form(55),
    wavelet: str = Form("haar"),
    threshold: float = Form(18),
    rank: int = Form(40),
) -> dict:
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Please choose an image first.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be smaller than 12 MB.")

    original_pil = _load_rgb_image(payload)
    original_array = np.asarray(original_pil, dtype=np.uint8)

    if technique == "dct":
        if not 10 <= quality <= 90:
            raise HTTPException(status_code=422, detail="DCT quality must be between 10 and 90.")
        processed_array = _dct_compress(original_array, quality)
    elif technique == "dwt":
        if not 0 <= threshold <= 60:
            raise HTTPException(status_code=422, detail="DWT threshold must be between 0 and 60.")
        processed_array = _dwt_compress(original_array, wavelet, threshold)
    else:
        if rank < 1:
            raise HTTPException(status_code=422, detail="SVD rank must be at least 1.")
        processed_array, retained_rank, maximum_rank = _svd_compress(original_array, rank)

    # WebP provides a compact downloadable image while retaining the selected
    # transform's reconstructed pixels as the source for encoding.
    output_name = f"{technique}_{uuid.uuid4().hex}.webp"
    output_path = OUTPUT_DIR / output_name
    processed_pil = Image.fromarray(processed_array, mode="RGB")
    processed_pil.save(output_path, format="WEBP", quality=88, method=6)

    # Measure the actual saved output, not only the pre-encoding array.
    with Image.open(output_path) as saved_image:
        saved_array = np.asarray(saved_image.convert("RGB"), dtype=np.uint8)

    metrics = _make_metrics(
        original_array,
        saved_array,
        len(payload),
        output_path.stat().st_size,
    )

    if technique == "dct":
        metrics["quality_setting"] = quality
    elif technique == "dwt":
        metrics["wavelet"] = wavelet
        metrics["threshold"] = threshold
    else:
        metrics["retained_rank"] = retained_rank
        metrics["maximum_rank"] = maximum_rank

    return {
        "technique": technique,
        "output_filename": output_name,
        "processed_url": f"/api/result/{output_name}",
        "metrics": metrics,
    }


@app.get("/api/result/{filename}")
def get_processed_image(filename: str) -> FileResponse:
    # Accept generated WebP names only; never allow arbitrary file paths.
    if Path(filename).name != filename or not filename.endswith(".webp"):
        raise HTTPException(status_code=404, detail="Result not found.")

    output_path = OUTPUT_DIR / filename
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="This result has expired or does not exist.")

    return FileResponse(output_path, media_type="image/webp")


@app.get("/")
def home_page() -> FileResponse:
    return FileResponse(ROOT_DIR / "index.html")


@app.get("/home")
def home_alias() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=308)


@app.get("/technique/{technique}")
def technique_page(technique: str):
    if technique not in {"dct", "dwt", "svd"}:
        raise HTTPException(status_code=404, detail="Technique not found.")
    return FileResponse(ROOT_DIR / "technique.html")


@app.get("/upload/{technique}")
def upload_page(technique: str):
    if technique not in {"dct", "dwt", "svd"}:
        raise HTTPException(status_code=404, detail="Technique not found.")
    return FileResponse(ROOT_DIR / "upload.html")


@app.get("/app.js")
def frontend_javascript() -> FileResponse:
    return FileResponse(ROOT_DIR / "app.js", media_type="application/javascript")


@app.get("/styles.css")
def frontend_stylesheet() -> FileResponse:
    return FileResponse(ROOT_DIR / "styles.css", media_type="text/css")


@app.get("/assets/{asset_path:path}")
def frontend_asset(asset_path: str) -> FileResponse:
    requested = (ASSETS_DIR / asset_path).resolve()
    try:
        requested.relative_to(ASSETS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Asset not found.")

    if not requested.is_file():
        raise HTTPException(status_code=404, detail="Asset not found.")
    return FileResponse(requested)


# Authentication pages are retired; old bookmarks return to the public homepage.
@app.get("/login")
@app.get("/signin")
def retired_auth_pages() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=308)


@app.get("/offline.html", include_in_schema=False)
def offline_page() -> FileResponse:
    return FileResponse(ROOT_DIR / "offline.html")


@app.get("/404.html", include_in_schema=False)
def not_found_page() -> FileResponse:
    return FileResponse(ROOT_DIR / "404.html", status_code=404)


@app.get("/service-worker.js", include_in_schema=False)
def service_worker() -> FileResponse:
    return FileResponse(
        ROOT_DIR / "service-worker.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


@app.exception_handler(404)
async def custom_not_found(request: Request, exc: Exception):
    # Keep API errors machine-readable; render the branded page for browser routes.
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=404, content={"detail": "The requested API endpoint was not found."})
    return FileResponse(ROOT_DIR / "404.html", status_code=404)


# ── Motion blur estimation and restoration ───────────────────────────────
def _motion_kernel(kind: str, length: int, angle: float) -> np.ndarray:
    """Build a normalized point-spread function for the selected motion model."""
    length = int(np.clip(length, 3, 45))
    radius = max(2, length // 2)
    side = 2 * radius + 1
    kernel = np.zeros((side, side), dtype=np.float64)
    center = radius

    if kind in {"linear", "uniform"}:
        theta = np.deg2rad(angle)
        xs = np.linspace(-radius, radius, length)
        weights = np.exp(-0.5 * (xs / max(radius / 2, 1)) ** 2) if kind == "linear" else np.ones_like(xs)
        for x, weight in zip(xs, weights):
            px = int(round(center + x * np.cos(theta)))
            py = int(round(center + x * np.sin(theta)))
            if 0 <= px < side and 0 <= py < side:
                kernel[py, px] += float(weight)
    elif kind == "circular":
        # A short arc is a local approximation of rotational camera motion.
        angles = np.linspace(-np.deg2rad(min(length * 2.0, 100)), np.deg2rad(min(length * 2.0, 100)), length * 3)
        for a in angles:
            px = int(round(center + radius * np.cos(a)))
            py = int(round(center + radius * np.sin(a)))
            if 0 <= px < side and 0 <= py < side:
                kernel[py, px] += 1.0
    elif kind == "non-uniform":
        # Smoothly curved trajectory approximates varying motion over exposure.
        t = np.linspace(-1.0, 1.0, length * 2)
        theta = np.deg2rad(angle)
        for value in t:
            x = value * radius
            y = 0.32 * radius * np.sin(np.pi * value)
            px = int(round(center + x * np.cos(theta) - y * np.sin(theta)))
            py = int(round(center + x * np.sin(theta) + y * np.cos(theta)))
            if 0 <= px < side and 0 <= py < side:
                kernel[py, px] += 1.0
    else:
        raise HTTPException(status_code=422, detail="Blur type must be linear, circular, uniform or non-uniform.")

    total = float(kernel.sum())
    if total <= 0:
        kernel[center, center] = 1.0
        total = 1.0
    return kernel / total


def _wiener_restore(image_array: np.ndarray, kernel: np.ndarray, strength: float) -> np.ndarray:
    """Restore RGB channels using a stabilized frequency-domain Wiener filter."""
    height, width, _ = image_array.shape
    padded_kernel = np.zeros((height, width), dtype=np.float64)
    kh, kw = kernel.shape
    padded_kernel[:kh, :kw] = kernel
    # Move the PSF centre to the FFT origin.
    padded_kernel = np.roll(padded_kernel, -(kh // 2), axis=0)
    padded_kernel = np.roll(padded_kernel, -(kw // 2), axis=1)
    transfer = np.fft.fft2(padded_kernel)
    conjugate = np.conj(transfer)
    denominator = (np.abs(transfer) ** 2) + max(0.0001, float(strength))
    restored = np.empty_like(image_array, dtype=np.float64)

    for channel in range(3):
        source = image_array[:, :, channel].astype(np.float64) / 255.0
        spectrum = np.fft.fft2(source)
        estimate = np.fft.ifft2((conjugate / denominator) * spectrum).real
        restored[:, :, channel] = estimate * 255.0

    return np.clip(np.rint(restored), 0, 255).astype(np.uint8)


def _estimate_kernel_length(image_array: np.ndarray, supplied_length: int) -> int:
    """Use the user's motion-length prior; retain a safe bounded estimate."""
    # A blind estimator is ill-posed for arbitrary natural images. This estimate
    # constrains the model to the requested prior instead of pretending certainty.
    shortest_side = min(image_array.shape[:2])
    if shortest_side < 5:
        raise HTTPException(status_code=422, detail="Motion restoration requires images at least 5 × 5 pixels.")
    return int(np.clip(supplied_length, 3, min(45, shortest_side - 1)))


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
        "method": "Wiener deconvolution (known-model estimate)",
        "processed_url": f"/api/result/{output_name}",
        "output_size_kb": round(output_path.stat().st_size / 1024, 2),
    }


@app.get("/motion-blur")
def motion_blur_page() -> FileResponse:
    return FileResponse(ROOT_DIR / "motion-blur.html")


@app.get("/motion-blur.css")
def motion_blur_stylesheet() -> FileResponse:
    return FileResponse(ROOT_DIR / "motion-blur.css", media_type="text/css")


@app.get("/motion-blur.js")
def motion_blur_javascript() -> FileResponse:
    return FileResponse(ROOT_DIR / "motion-blur.js", media_type="application/javascript")
