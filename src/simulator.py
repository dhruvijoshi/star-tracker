import numpy as np
import cv2

def render_starfield(camera, catalog):
    # 8-bit grayscale image for robust display
    img = np.zeros((camera.height, camera.width), dtype=np.uint8)

    # Star directions and magnitudes
    if all(c in catalog.columns for c in ('x', 'y', 'z')):
        dirs = catalog[['x', 'y', 'z']].values.astype(np.float64)
    elif all(c in catalog.columns for c in ('ux', 'uy', 'uz')):
        dirs = catalog[['ux', 'uy', 'uz']].values.astype(np.float64)
    else:
        raise RuntimeError('Catalog missing unit-vector columns (x,y,z or ux,uy,uz)')

    mags = catalog['mag'].values.astype(np.float64)

    coords, mask = camera.project(dirs, return_mask=True)
    if coords.size == 0 or not np.any(mask):
        return img

    # Draw stars
    drawn = 0
    for (x, y), mag in zip(coords, mags[mask]):
        if 0 <= x < camera.width and 0 <= y < camera.height:
            
            rel = pow(10.0, -0.4 * (mag - 2.0))
            brightness = int(np.clip(np.sqrt(rel) * 255.0, 80, 255))
            size = 3 if mag > 5.5 else 4 if mag > 4.5 else 5
            cv2.circle(img, (int(x), int(y)), size, brightness, -1)
            drawn += 1

    if drawn == 0 and coords.size > 0:
        cx, cy = camera.width / 2.0, camera.height / 2.0
        d2 = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
        idx = int(np.argmin(d2))
        x, y = coords[idx]
        mag = float(mags[mask][idx])
        rel = pow(10.0, -0.4 * (mag - 2.0))
        brightness = int(np.clip(np.sqrt(rel) * 255.0, 120, 255))
        size = 5
        x_i = int(np.clip(round(x), 0, camera.width - 1))
        y_i = int(np.clip(round(y), 0, camera.height - 1))
        cv2.circle(img, (x_i, y_i), size, brightness, -1)

    # Gaussian blur for more natural appearance
    if img.max() > 0:
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return img