import numpy as np
import cv2
from typing import List, Tuple


def find_centroids(
    image: np.ndarray,
    threshold: float = None,
    min_area: int = 2,
    max_area: int = 300,
    max_stars: int = 50,
) -> List[Tuple[float, float, float]]:
    """
    Detect star centroids in a grayscale image using connected-component analysis.

    Stars are located by thresholding and then computing an intensity-weighted
    centroid (centre of mass) for each blob — more accurate than geometric centre.

    Args:
        image:      2-D uint8 grayscale image
        threshold:  pixel value above which a pixel is considered a star.
                    Defaults to mean + 3 * std of the image (auto-threshold).
        min_area:   minimum blob area in pixels (rejects single-pixel noise)
        max_area:   maximum blob area in pixels (rejects extended artefacts)
        max_stars:  return at most this many centroids

    Returns:
        List of (x, y, flux) tuples sorted brightest first,
        where x, y are sub-pixel centroid positions in image coordinates
        and flux is the summed pixel intensity of the blob.
    """
    if image.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image, got shape " + str(image.shape))

    img = image.astype(np.float32)

    # Auto-threshold: mean + 3 sigma keeps only bright outliers
    if threshold is None:
        mu = float(img.mean())
        sigma = float(img.std())
        threshold = mu + 3.0 * sigma

    _, binary = cv2.threshold(img, float(threshold), 255.0, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    centroids: List[Tuple[float, float, float]] = []
    for label in range(1, num_labels):  # label 0 = background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        mask = labels == label
        ys, xs = np.nonzero(mask)
        intensities = img[mask]
        flux = float(intensities.sum())
        if flux == 0:
            continue

        # Intensity-weighted centroid
        cx = float(np.dot(xs, intensities) / flux)
        cy = float(np.dot(ys, intensities) / flux)
        centroids.append((cx, cy, flux))

    centroids.sort(key=lambda t: -t[2])
    return centroids[:max_stars]
