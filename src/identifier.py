"""
Triangle-based lost-in-space star identification.

Algorithm overview
------------------
1. A subset of bright catalog stars is selected and all pairwise angular
   separations are precomputed once at construction time.
2. For a set of detected centroids, candidate triplets (A, B, C) are formed
   from the brightest detected stars.
3. The three inter-star angles in the image (derived from pixel distances and
   the camera focal length) are looked up in the sorted pair database.
4. A consistent catalog triplet (p, q, r) is returned when the three angles
   agree within tolerance.

References
----------
Mortari, D. et al. (2004). "The Pyramid Star Identification Technique."
  Navigation, 51(3), 171-183.
"""

import itertools
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


class StarIdentifier:
    """
    Lost-in-space star identifier using the triangle algorithm.

    Parameters
    ----------
    catalog : pd.DataFrame
        Star catalog containing unit-vector columns ('ux','uy','uz' or 'x','y','z')
        and a 'mag' column.
    fov_deg : float
        Camera horizontal field of view in degrees.
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    mag_limit : float
        Only catalog stars brighter than this magnitude are used for matching.
        Smaller values → fewer stars → faster but may miss faint-field images.
    angle_tol_deg : float
        Angular tolerance (degrees) when comparing measured and catalog angles.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        fov_deg: float,
        img_width: int,
        img_height: int,
        mag_limit: float = 5.5,
        angle_tol_deg: float = 0.4,
    ) -> None:
        # Select bright stars and sort by magnitude
        bright = catalog[catalog["mag"] <= mag_limit].copy()
        bright = bright.sort_values("mag").reset_index(drop=True)

        if len(bright) < 3:
            raise ValueError(
                f"Too few bright stars (mag ≤ {mag_limit}): found {len(bright)}. "
                "Try increasing mag_limit."
            )

        vec_cols = (
            ["ux", "uy", "uz"]
            if "ux" in bright.columns
            else ["x", "y", "z"]
        )
        self.vectors: np.ndarray = bright[vec_cols].values.astype(np.float64)  # (N,3)
        self.bright_catalog: pd.DataFrame = bright  # kept for downstream use

        N = len(self.vectors)
        self.N = N

        # ------------------------------------------------------------------
        # Precompute N×N angle matrix (radians) — O(N²) memory but fast lookups
        # For N≤2000 (mag≤5.5 gives ~1700 HYG stars) this is ~26 MB, acceptable.
        # ------------------------------------------------------------------
        dots = np.clip(self.vectors @ self.vectors.T, -1.0, 1.0)
        self._angle_matrix: np.ndarray = np.arccos(dots)  # (N, N)

        # Sorted flat pair list for binary-search lookup
        ii, jj = np.triu_indices(N, k=1)
        pair_angles = self._angle_matrix[ii, jj]
        order = np.argsort(pair_angles)
        self._pair_angles: np.ndarray = pair_angles[order]
        self._pair_i: np.ndarray = ii[order].astype(np.int32)
        self._pair_j: np.ndarray = jj[order].astype(np.int32)

        # Camera geometry
        fov_rad = np.deg2rad(fov_deg)
        self.focal: float = (img_width / 2.0) / np.tan(fov_rad / 2.0)
        self.angle_tol: float = np.deg2rad(angle_tol_deg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pixel_to_angle(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Convert pixel separation between two points to angular separation (radians)."""
        d = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        return float(np.arctan2(d, self.focal))

    def _pairs_in_range(self, angle: float) -> Tuple[int, int]:
        """Return (lo, hi) slice indices of self._pair_angles within ±tol of angle."""
        lo = int(np.searchsorted(self._pair_angles, angle - self.angle_tol, side="left"))
        hi = int(np.searchsorted(self._pair_angles, angle + self.angle_tol, side="right"))
        return lo, hi

    def _try_triplet(
        self,
        a: int,
        b: int,
        c: int,
        p: int,
        q: int,
        θ_ac: float,
        θ_bc: float,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Given centroid triplet (a,b,c) mapped to catalog stars (p,q,?),
        search for catalog star r satisfying angle(p,r)≈θ_ac and angle(q,r)≈θ_bc.
        Returns match list on success, None otherwise.
        """
        angles_from_p = self._angle_matrix[p]
        candidates = np.where(np.abs(angles_from_p - θ_ac) < self.angle_tol)[0]
        for r in candidates:
            if r == p or r == q:
                continue
            if abs(self._angle_matrix[q, r] - θ_bc) < self.angle_tol:
                return [(a, int(p)), (b, int(q)), (c, int(r))]
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def identify(
        self,
        centroids: List[Tuple[float, float, float]],
        max_tries: int = 200,
    ) -> List[Tuple[int, int]]:
        """
        Match detected centroids to catalog stars.

        Parameters
        ----------
        centroids : list of (x, y, flux)
            Detected star positions sorted brightest first (output of
            centroid.find_centroids).
        max_tries : int
            Maximum number of centroid triplets to examine before giving up.

        Returns
        -------
        List of (centroid_index, identifier_internal_star_index) pairs.
        Each entry says "centroid #i corresponds to bright-catalog star #j".
        Empty list means identification failed.
        """
        n = min(len(centroids), 12)
        if n < 3:
            return []

        tries = 0
        for a, b, c in itertools.combinations(range(n), 3):
            if tries >= max_tries:
                break
            tries += 1

            pa = centroids[a][:2]
            pb = centroids[b][:2]
            pc = centroids[c][:2]

            θ_ab = self._pixel_to_angle(pa, pb)
            θ_ac = self._pixel_to_angle(pa, pc)
            θ_bc = self._pixel_to_angle(pb, pc)

            lo, hi = self._pairs_in_range(θ_ab)
            if hi <= lo:
                continue

            for idx in range(lo, hi):
                p = int(self._pair_i[idx])
                q = int(self._pair_j[idx])

                match = self._try_triplet(a, b, c, p, q, θ_ac, θ_bc)
                if match is not None:
                    return match

                match = self._try_triplet(a, b, c, q, p, θ_ac, θ_bc)
                if match is not None:
                    return match

        return []
