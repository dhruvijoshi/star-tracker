"""
Attitude determination from matched star pairs using Wahba's SVD method.

Returns a rotation matrix R such that:
    v_camera = R @ v_catalog
i.e. R rotates catalog (inertial) frame unit vectors into the camera (body) frame.
"""

import numpy as np
from typing import List, Optional, Tuple


def wahba(
    body_vectors: np.ndarray,
    ref_vectors: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Wahba's problem: optimal rotation matrix via SVD.

    Minimises  Σ_i w_i * ‖v_body_i  −  R @ v_ref_i‖²

    Parameters
    ----------
    body_vectors : (N, 3) unit vectors in the camera/body frame.
    ref_vectors  : (N, 3) corresponding unit vectors in the inertial frame.
    weights      : (N,) non-negative weights (default: uniform).

    Returns
    -------
    3×3 rotation matrix R such that v_body ≈ R @ v_ref.
    """
    if weights is None:
        weights = np.ones(len(body_vectors), dtype=np.float64)

    # Attitude profile matrix B = Σ w_i * v_body_i * v_ref_i^T
    B = (weights[:, None, None] * body_vectors[:, :, None] * ref_vectors[:, None, :]).sum(axis=0)

    U, _, Vt = np.linalg.svd(B)

    # Ensure proper rotation (det = +1), not an improper rotation
    d = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.diag([1.0, 1.0, d])

    return U @ D @ Vt


def attitude_from_matches(
    matches: List[Tuple[int, int]],
    centroids: List[Tuple[float, float, float]],
    catalog_vectors: np.ndarray,
    focal: float,
    img_width: int,
    img_height: int,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Compute the camera attitude from (centroid, catalog_star) correspondences.

    Each centroid is back-projected to a unit vector in the camera frame using
    the pinhole model, then Wahba's SVD solver finds the best rotation R.

    Parameters
    ----------
    matches          : list of (centroid_idx, catalog_star_idx) from StarIdentifier.
    centroids        : list of (x, y, flux) — full list passed to the identifier.
    catalog_vectors  : (M, 3) unit vectors for the identifier's bright catalog
                       (pass identifier.vectors).
    focal            : camera focal length in pixels.
    img_width/height : image dimensions (used to locate the principal point).

    Returns
    -------
    (R, residual_deg)
        R             — 3×3 rotation matrix (camera = R @ inertial).
        residual_deg  — mean angular reprojection error in degrees
                        (lower is better; < 0.5° is good for a star tracker).
        Returns (None, inf) if fewer than 2 matches are provided.
    """
    if len(matches) < 2:
        return None, float("inf")

    cx = img_width / 2.0
    cy = img_height / 2.0

    body_vecs = []
    ref_vecs = []

    for ci, si in matches:
        x, y, _ = centroids[ci]
        # Back-project: pixel → normalised image coords → unit vector
        v_body = np.array([x - cx, y - cy, focal], dtype=np.float64)
        v_body /= np.linalg.norm(v_body)
        body_vecs.append(v_body)
        ref_vecs.append(catalog_vectors[si])

    body_vecs = np.array(body_vecs)  # (N, 3)
    ref_vecs = np.array(ref_vecs)    # (N, 3)

    R = wahba(body_vecs, ref_vecs)

    # Compute mean angular residual
    reprojected = (R @ ref_vecs.T).T  # (N, 3)
    cos_angles = np.clip(
        np.einsum("ij,ij->i", body_vecs, reprojected), -1.0, 1.0
    )
    residual_deg = float(np.degrees(np.arccos(cos_angles)).mean())

    return R, residual_deg


def rotation_to_radec(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract the RA / Dec / Roll of the camera boresight from a rotation matrix.

    The boresight is the +Z axis of the camera frame.  In the inertial frame
    it corresponds to the column of R^T that maps to [0, 0, 1] in body frame,
    i.e. the third row of R (since v_body = R @ v_ref  →  z_body = R[2,:]).

    Returns
    -------
    (ra_deg, dec_deg, roll_deg)
        ra_deg  : Right Ascension  [0, 360)
        dec_deg : Declination      [-90, 90]
        roll_deg: camera roll about the boresight  [0, 360)
    """
    # Boresight direction in inertial frame: R^T @ [0,0,1] = R[:,2]
    boresight = R.T @ np.array([0.0, 0.0, 1.0])

    x, y, z = boresight
    ra_deg = float(np.degrees(np.arctan2(y, x)) % 360.0)
    dec_deg = float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))))

    # Roll: angle between camera +Y and the North direction projected onto
    # the image plane (standard definition used in spacecraft).
    north = np.array([0.0, 0.0, 1.0])
    north_perp = north - np.dot(north, boresight) * boresight
    n = np.linalg.norm(north_perp)
    if n > 1e-10:
        north_perp /= n
        cam_up = R.T @ np.array([0.0, 1.0, 0.0])
        cos_roll = np.clip(np.dot(cam_up, north_perp), -1.0, 1.0)
        sin_roll = np.dot(np.cross(north_perp, cam_up), boresight)
        roll_deg = float(np.degrees(np.arctan2(sin_roll, cos_roll)) % 360.0)
    else:
        roll_deg = 0.0

    return ra_deg, dec_deg, roll_deg
