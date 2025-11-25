from src.catalog import load_catalog
from src.camera import Camera
from src.simulator import render_starfield
import cv2
import numpy as np

def main():
    # Load catalog 
    catalog = load_catalog("data/catalog/hygdata_v42_compact.csv", mag_limit=None)

    # Camera resolution and FOV
    cam = Camera(width=1280, height=720, fov_deg=60)

    # Point toward a randomly selected star
    xyz = catalog[['x', 'y', 'z']].values
    idx = np.random.randint(0, len(xyz))
    vx, vy, vz = map(float, xyz[idx])
    ra_deg = (np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0
    dec_deg = float(np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0))))
    cam.point_at(ra_deg=ra_deg, dec_deg=dec_deg)

    # Render once and retry if no stars are visible
    img = render_starfield(cam, catalog)
    tries = 0
    while tries < 4 and img.max() == 0:
        idx = np.random.randint(0, len(xyz))
        vx, vy, vz = map(float, xyz[idx])
        ra_deg = (np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0
        dec_deg = float(np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0))))
        cam.point_at(ra_deg=ra_deg, dec_deg=dec_deg)
        img = render_starfield(cam, catalog)
        tries += 1

    cv2.imshow("Synthetic Starfield", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
        