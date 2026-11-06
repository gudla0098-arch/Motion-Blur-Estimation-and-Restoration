# Motion Blur Estimation and Restoration

A standalone Digital Image Processing project for modeling motion blur and restoring images with point-spread functions (PSFs) and Wiener deconvolution.

## Features

- Upload PNG, JPG/JPEG, or WEBP images (12 MB maximum; 4 million decoded pixels).
- Select linear, circular, uniform, or non-uniform motion PSF models.
- Configure motion length, direction angle, and Wiener regularization strength.
- Preview and download restored WEBP output.
- FastAPI backend with a browser-based interface.

## Important scientific note

This implementation uses user-supplied motion parameters to construct a PSF prior and applies Wiener deconvolution. It is **model-based restoration**, not blind motion-kernel estimation from arbitrary images. Circular and non-uniform PSFs are simplified approximations. Restoration quality depends on how closely the selected model matches the actual blur and on noise and information loss in the input image.

## Project layout

```text
Motion-Blur-Estimation-and-Restoration/
├── api/index.py
├── backend/main.py
├── motion-blur.html
├── motion-blur.css
├── motion-blur.js
├── requirements.txt
├── vercel.json
└── README.md
```

## Run locally

Use Python 3.12.

```bash
git clone https://github.com/gudla0098-arch/Motion-Blur-Estimation-and-Restoration.git
cd Motion-Blur-Estimation-and-Restoration
python -m venv .venv
```

Activate the environment, then install dependencies and run:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/motion-blur`.

## API

`POST /api/motion-blur` accepts `multipart/form-data`:

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `image` | File | Required | PNG, JPG/JPEG, WEBP |
| `blur_type` | String | `linear` | `linear`, `circular`, `uniform`, `non-uniform` |
| `length` | Integer | `18` | Motion length in pixels, 3–45 and bounded by image dimensions |
| `angle` | Number | `0` | Motion direction in degrees, 0–180 |
| `strength` | Number | `0.008` | Wiener regularization, 0.001–0.05 |

Successful responses include the selected model, motion length, angle, kernel size, method, `processed_url`, and output file size. Fetch the result using `GET /api/result/{filename}`.

## Deployment

Import this repository into Vercel. The included `vercel.json` configures the Python function entry point at `api/index.py`.
