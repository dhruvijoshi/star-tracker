# Star Tracker

**Live demo:** [startracker.streamlit.app](https://startracker.streamlit.app/)

A star tracker simulator for spacecraft and satellites. It renders a synthetic night-sky image using a real star catalog, then runs the full attitude-determination pipeline on that image — the same pipeline a real flight star tracker uses.

**Star catalog:** [HYG v4.2](https://www.astronexus.com/projects/hyg) (Hipparcos / Yale / Gliese, ~120 000 stars)

---

## What is a Star Tracker?

A star tracker is an optical sensor mounted on a spacecraft. It captures a single image of the star field and answers one question: **which direction is the camera pointing?**

The answer is called the **attitude** — a 3-axis orientation expressed as Right Ascension (RA), Declination (Dec), and Roll angle. This is **not** a position fix. Stars are too far away to reveal where you are in space; they only reveal which way you are facing. Spacecraft combine star tracker data with GPS or orbit propagators to get a full 6-DOF state.

### Lost-in-space vs. track mode

A star tracker can operate in two modes:

- **Lost-in-space (LIS)** — zero prior knowledge of orientation. The algorithm identifies stars purely from their pattern, with no initial guess. This is the harder and more important case.
- **Track mode** — a rough prior attitude is known and refined frame-to-frame.

This project implements **lost-in-space** using the triangle algorithm.

---

## How It Works

```
Star catalog (HYG v4.2)
        │
        ▼
Synthetic image rendered by projecting catalog stars
through a pinhole camera model with realistic sensor noise
        │
        ▼
Centroid detection — finds sub-pixel star positions in the image
        │
        ▼
Triangle identification (lost-in-space) — matches star triplets
to catalog triplets by comparing inter-star angular separations
        │
        ▼
Attitude solver (Wahba / SVD) — finds the optimal rotation matrix
that maps catalog unit vectors onto the detected image vectors
        │
        ▼
RA / Dec / Roll + residual error
```

### Modules

| File | Responsibility |
|------|---------------|
| `src/preprocess.py` | Cleans the raw HYG CSV; outputs compact catalog with unit vectors and an optional KD-tree |
| `src/catalog.py` | Loads the compact catalog at runtime, applies magnitude filtering, exposes `x/y/z` unit vectors |
| `src/camera.py` | Pinhole camera model: focal length, intrinsic matrix, optional lens distortion, `point_at()` and `project()` |
| `src/simulator.py` | Renders a synthetic star field with supersampling, Gaussian blur, and configurable sensor noise (hot pixels, cosmic rays, vignetting, bloom, twinkling) |
| `src/centroid.py` | Detects stars via connected-component analysis; returns intensity-weighted sub-pixel centroids |
| `src/identifier.py` | Triangle-based lost-in-space identifier; precomputes an N×N inter-star angle table and matches detected triplets to catalog triplets |
| `src/attitude.py` | Wahba's SVD attitude solver (`wahba`, `attitude_from_matches`, `rotation_to_radec`) |
| `src/pipeline.py` | Ties the pipeline together: `solve(image, …) → AttitudeResult` |
| `main.py` | CLI entry point: renders a random synthetic image, solves attitude, prints result, displays image |
| `app.py` | Streamlit web UI: generate synthetic images or upload your own, with interactive noise controls and a live identified-star table |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `opencv-python`, `pandas`, `scipy`, `streamlit`

### 2. Preprocess the catalog

Run once to generate the data files used at runtime:

```bash
python src/preprocess.py --compact --save-npz data/catalog/hygdata_v42_compact_mag8_dist200.npz --save-tree data/catalog/hygdata_v42_compact_mag8_dist200_tree.pkl
```

This creates:

| File | Description |
|------|-------------|
| `data/catalog/hygdata_v42_clean.csv` | Cleaned full catalog |
| `data/catalog/hygdata_v42_compact.csv` | Filtered catalog (mag ≤ 8, dist ≤ 200 pc) with unit vectors |
| `data/catalog/hygdata_v42_compact_mag8_dist200.npz` | Compressed arrays: ids, ra/dec, mag, dist, ux/uy/uz |
| `data/catalog/hygdata_v42_compact_mag8_dist200_tree.pkl` | KD-tree for fast nearest-star queries |

### 3. Run

**Streamlit web app (recommended):**

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`. Choose a pointing direction (or randomize), adjust noise parameters, and hit **Generate & Solve** to render a synthetic star field and run the full attitude pipeline. You can also upload your own star image to solve.

**CLI:**

```bash
python main.py
```

Picks a random direction, renders a synthetic star field, solves the attitude, and prints the result:

```
Loading catalog...  24330 stars loaded.
Camera pointed at RA=142.57°  Dec=-33.21°
Rendering synthetic image...
  Saved → data/images/synthetic.png
Building star identifier...  1712 bright stars indexed.
Solving attitude...

Attitude solved
  RA       142.5831°
  Dec      -33.2044°
  Roll      87.3120°
  Residual  0.1823°
  Stars     9 matched / 31 detected
```

An OpenCV window shows the rendered image. Press any key to close.

---

## Noise Parameters

`NoiseParams` in `src/simulator.py` controls the sensor simulation:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gaussian_sigma` | 0.0 | Read noise / dark current (σ in pixel intensity units) |
| `hot_pixel_prob` | 0.0 | Fraction of permanently saturated pixels |
| `cosmic_ray_prob` | 0.0 | Rate of cosmic-ray streak events |
| `vignette_strength` | 0.0 | Lens vignetting (0 = off, 1 = full darkening at corners) |
| `bloom_threshold` | 0.8 | Relative brightness above which bloom spreads |
| `bloom_intensity` | 0.0 | Bloom spread strength |
| `twinkle_amount` | 0.0 | Atmospheric scintillation amplitude |
| `twinkle_speed` | 1.0 | Scintillation frequency |

---

## Attitude Output Reference

| Field | Description |
|-------|-------------|
| `ra_deg` | Right Ascension of the camera boresight (0–360°) |
| `dec_deg` | Declination of the camera boresight (−90–90°) |
| `roll_deg` | Camera roll about the boresight (0–360°) |
| `residual_deg` | Mean angular reprojection error across matched stars — values below 0.5° indicate a good solution |
| `n_matches` | Number of stars successfully matched to the catalog |
| `n_centroids` | Total number of star candidates detected in the image |
| `success` | `True` if attitude was solved; `False` if identification or solver failed |
