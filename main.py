import os
import time
import cv2
import numpy as np

from src.catalog import load_catalog
from src.camera import Camera
from src.simulator import render_starfield, NoiseParams
from src.identifier import StarIdentifier
from src import pipeline

IMAGE_SAVE_DIR = "data/images"
CATALOG_PATH   = "data/catalog/hygdata_v42_compact.csv"
IMG_WIDTH      = 1280
IMG_HEIGHT     = 720
FOV_DEG        = 60.0
MAG_LIMIT      = 8.0
ID_MAG_LIMIT   = 5.5


def main():
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

    # --- Catalog ---
    print("Loading catalog...")
    catalog = load_catalog(CATALOG_PATH, mag_limit=MAG_LIMIT)
    print(f"  {len(catalog)} stars loaded.")

    # --- Camera pointed at a random catalog star ---
    cam = Camera(width=IMG_WIDTH, height=IMG_HEIGHT, fov_deg=FOV_DEG)
    xyz = catalog[["x", "y", "z"]].values
    idx = np.random.randint(0, len(xyz))
    vx, vy, vz = map(float, xyz[idx])
    ra  = float((np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0)
    dec = float(np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0))))
    cam.point_at(ra_deg=ra, dec_deg=dec)
    print(f"  Camera pointed at RA={ra:.2f}°  Dec={dec:.2f}°")

    # --- Render synthetic image ---
    noise_params = NoiseParams(
        gaussian_sigma=5.0,
        hot_pixel_prob=0.0001,
        cosmic_ray_prob=0.0005,
        vignette_strength=0.3,
        bloom_threshold=0.8,
        bloom_intensity=0.3,
        twinkle_amount=0.2,
        twinkle_speed=1.0,
    )
    print("Rendering synthetic image...")
    img = render_starfield(camera=cam, catalog=catalog, noise_params=noise_params)
    tries = 0
    while tries < 4 and img.max() == 0:
        idx = np.random.randint(0, len(xyz))
        vx, vy, vz = map(float, xyz[idx])
        ra  = float((np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0)
        dec = float(np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0))))
        cam.point_at(ra_deg=ra, dec_deg=dec)
        img = render_starfield(camera=cam, catalog=catalog, noise_params=noise_params)
        tries += 1

    out_path = os.path.join(IMAGE_SAVE_DIR, f"synthetic_{int(time.time())}.png")
    cv2.imwrite(out_path, img)
    print(f"  Saved → {out_path}")

    # --- Build identifier ---
    print("Building star identifier...")
    identifier = StarIdentifier(
        catalog=catalog,
        fov_deg=FOV_DEG,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        mag_limit=ID_MAG_LIMIT,
        angle_tol_deg=0.4,
    )
    print(f"  {identifier.N} bright stars indexed.")

    # --- Solve attitude ---
    print("Solving attitude...")
    result = pipeline.solve(img, identifier, cam.focal, IMG_WIDTH, IMG_HEIGHT)

    print()
    if result.success:
        print("Attitude solved")
        print(f"  RA       {result.ra_deg:.4f}°")
        print(f"  Dec      {result.dec_deg:.4f}°")
        print(f"  Roll     {result.roll_deg:.4f}°")
        print(f"  Residual {result.residual_deg:.4f}°")
        print(f"  Stars    {result.n_matches} matched / {result.n_centroids} detected")
    else:
        print("Attitude solution failed")
        print(f"  {result.n_centroids} centroids detected, {result.n_matches} matched.")

    # --- Display ---
    cv2.imshow("Synthetic Starfield", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
