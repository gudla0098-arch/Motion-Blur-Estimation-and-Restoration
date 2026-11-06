<div align="center">

# Motion Blur Estimation & Restoration

**A browser-based Digital Image Processing lab for model-based motion blur restoration.**

[![CI](https://github.com/gudla0098-arch/Motion-Blur-Estimation-and-Restoration/actions/workflows/smoke-check.yml/badge.svg?branch=main)](https://github.com/gudla0098-arch/Motion-Blur-Estimation-and-Restoration/actions/workflows/smoke-check.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Open Live Demo**](https://motion-blur-estimation-and-restoration-git-main-raju-d9aa.vercel.app) · [Report a Bug](https://github.com/gudla0098-arch/Motion-Blur-Estimation-and-Restoration/issues) · [Request a Feature](https://github.com/gudla0098-arch/Motion-Blur-Estimation-and-Restoration/issues)

</div>

---

## Overview

Motion blur appears when a camera or subject moves while an image is being captured. This project provides an interactive workflow to model that movement with a **point-spread function (PSF)** and attempt image recovery through **Wiener deconvolution**.

Upload an image, choose a blur model, tune the motion parameters, and compare or download the restored result.

## Features

- **Image input:** PNG, JPG/JPEG, and WEBP uploads (up to 12 MB).
- **Four PSF models:** linear, circular, uniform, and non-uniform approximations.
- **Interactive controls:** motion length, direction angle, and restoration strength.
- **Restoration preview:** compare the original and processed image side by side.
- **Download:** save the restored image as WEBP.
- **API:** FastAPI endpoint for image processing and generated-result retrieval.
- **CI checks:** GitHub Actions verifies Python compilation, JavaScript syntax, routes, kernel normalization, and a restoration smoke test.

## Demo

**Live application:** https://motion-blur-estimation-and-restoration-git-main-raju-d9aa.vercel.app

> The demo may be unavailable while a deployment is building or if the hosting service is experiencing an issue.

## How it works

```text
Upload image
    |
    v
Choose a motion PSF model
    |
    v
Set motion length, angle and regularization
    |
    v
Build normalized point-spread function (PSF)
    |
    v
Apply frequency-domain Wiener deconvolution
    |
    v
Preview and download restored image
```

### Processing models

| Model | Approximation |
|---|---|
| Linear | Directional straight-line motion with a weighted kernel |
| Circular | Approximate rotational movement around the image centre |
| Uniform | Constant-weight straight-line motion |
| Non-uniform | Curved, spatially varying motion approximation |

### Scientific scope and limitations

This application is **model-based restoration**, not blind estimation of an unknown blur kernel. The user selects a model and supplies motion parameters; the application constructs a PSF prior and applies Wiener deconvolution.

The circular and non-uniform models are simplified approximations. Restoration cannot reliably recover information that was lost during capture, and results depend on the accuracy of the selected model, image noise, and blur severity. The output should be treated as an estimate, not a guaranteed reconstruction of the original scene.

## Technology stack

| Layer | Tools |
|---|---|
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Python, FastAPI |
| Image processing | NumPy, Pillow, FFT-based Wiener filtering |
| Hosting | Vercel |
| Quality checks | GitHub Actions |

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── smoke-check.yml     # Automated smoke checks
├── api/
│   └── index.py                # Vercel ASGI entry point
├── backend/
│   └── main.py                 # FastAPI routes and image processing
├── motion-blur.html            # Main application page
├── motion-blur.css             # Motion Blur page styles
├── motion-blur.js              # Upload, controls and API interaction
├── styles.css                  # Shared site styles
├── requirements.txt            # Python dependencies
├── vercel.json                 # Vercel deployment configuration
├── LICENSE
└── README.md
```

## Run locally

### 1. Requirements

- Python 3.12
- Git

### 2. Clone the repository

```bash
git clone https://github.com/gudla0098-arch/Motion-Blur-Estimation-and-Restoration.git
cd Motion-Blur-Estimation-and-Restoration
```

### 3. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Start the server

```bash
uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000/motion-blur** in your browser.

Interactive API documentation is available at **http://127.0.0.1:8000/docs**.

## API reference

### `POST /api/motion-blur`

Accepts `multipart/form-data`.

| Field | Type | Default | Description |
|---|---|---:|---|
| `image` | File | Required | PNG, JPG/JPEG, or WEBP image |
| `blur_type` | String | `linear` | `linear`, `circular`, `uniform`, or `non-uniform` |
| `length` | Integer | `18` | Motion length in pixels (3–45, bounded by image dimensions) |
| `angle` | Number | `0` | Motion direction in degrees (0–180) |
| `strength` | Number | `0.008` | Wiener regularization strength (0.001–0.05) |

A successful response includes the selected model and processing parameters, plus a `processed_url` for the generated output.

### `GET /api/result/{filename}`

Returns a generated WEBP result by its server-generated filename.

## Deployment

This repository is configured for Vercel through `vercel.json`, with `api/index.py` as the Python ASGI entry point.

1. Import `gudla0098-arch/Motion-Blur-Estimation-and-Restoration` into Vercel.
2. Keep the repository root as the project root.
3. Deploy using the included configuration.
4. Confirm the deployment and test image upload, processing, and download from the live URL.

Generated files are written to the temporary output directory configured by the backend. On serverless hosting, temporary storage is not permanent; generated results should be downloaded while they are available.

## Quality checks

The GitHub Actions workflow runs on pushes and pull requests targeting `main`. It currently checks:

- Python source compilation.
- JavaScript syntax.
- Expected API routes.
- PSF normalization for all four model types.
- Output dimensions from a small Wiener-restoration smoke test.

These are automated smoke checks, not a substitute for full browser testing against a live deployment.

## Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a descriptive branch: `git checkout -b feature/your-change`.
3. Make a focused change and test it locally.
4. Commit with a clear message.
5. Open a pull request describing the change and its verification.

Please include steps to reproduce bugs and, where appropriate, a small sample image that is safe to share.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built as a Digital Image Processing project.**

</div>
