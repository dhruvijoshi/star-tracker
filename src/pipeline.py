from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from src.centroid import find_centroids
from src.identifier import StarIdentifier
from src.attitude import attitude_from_matches, rotation_to_radec


@dataclass
class AttitudeResult:
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    roll_deg: float = 0.0
    residual_deg: float = float("inf")
    n_matches: int = 0
    n_centroids: int = 0
    centroids: List[Tuple[float, float, float]] = field(default_factory=list)
    matches: List[Tuple[int, int]] = field(default_factory=list)
    success: bool = False


def solve(
    image: np.ndarray,
    identifier: StarIdentifier,
    focal: float,
    img_width: int,
    img_height: int,
) -> AttitudeResult:
    """
    Run the full attitude pipeline on a single grayscale image.

    Parameters
    ----------
    image      : uint8 grayscale (H, W) — from render_starfield or cv2.imread
    identifier : built StarIdentifier (holds bright-catalog vectors + angle table)
    focal      : camera focal length in pixels  (Camera.focal or (w/2)/tan(fov/2))
    img_width, img_height : image dimensions in pixels

    Returns
    -------
    AttitudeResult — check .success before reading ra/dec/roll
    """
    centroids = find_centroids(image)
    n = len(centroids)

    if n < 3:
        return AttitudeResult(n_centroids=n, centroids=centroids)

    matches = identifier.identify(centroids)
    if not matches:
        return AttitudeResult(n_centroids=n, centroids=centroids)

    R, residual = attitude_from_matches(
        matches, centroids, identifier.vectors, focal, img_width, img_height
    )
    if R is None:
        return AttitudeResult(
            n_centroids=n, centroids=centroids,
            n_matches=len(matches), matches=matches,
        )

    ra, dec, roll = rotation_to_radec(R)
    return AttitudeResult(
        ra_deg=ra, dec_deg=dec, roll_deg=roll,
        residual_deg=residual,
        n_matches=len(matches), n_centroids=n,
        centroids=centroids, matches=matches,
        success=True,
    )
