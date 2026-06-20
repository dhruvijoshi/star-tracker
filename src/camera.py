import numpy as np
from typing import Tuple, Optional, Union


class Camera:
    def __init__(self, width: int = 640, height: int = 480, fov_deg: float = 60.0,
                 distortion: Optional[dict] = None):
        self.width = int(width)
        self.height = int(height)
        self.fov = np.deg2rad(fov_deg)
        self.focal = (self.width / 2.0) / np.tan(self.fov / 2.0)
        self.R = np.eye(3)

        self.K = np.array([
            [self.focal, 0, self.width / 2.0],
            [0, self.focal, self.height / 2.0],
            [0, 0, 1.0]
        ])

        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        if distortion:
            self.dist_coeffs[0] = distortion.get('k1', 0.0)
            self.dist_coeffs[1] = distortion.get('k2', 0.0)
            self.dist_coeffs[2] = distortion.get('p1', 0.0)
            self.dist_coeffs[3] = distortion.get('p2', 0.0)
            self.dist_coeffs[4] = distortion.get('k3', 0.0)

    def project(self, dirs: np.ndarray, return_mask: bool = False,
                apply_distortion: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # Stars are at infinity — only rotation matters
        dirs_cam = dirs @ self.R

        mask = dirs_cam[:, 2] > 1e-8
        if not np.any(mask):
            if return_mask:
                return np.zeros((0, 2), dtype=np.float32), mask
            return np.zeros((0, 2), dtype=np.float32)

        x = dirs_cam[mask, 0] / dirs_cam[mask, 2]
        y = dirs_cam[mask, 1] / dirs_cam[mask, 2]

        if apply_distortion and np.any(self.dist_coeffs != 0):
            r2 = x*x + y*y
            r4 = r2 * r2
            r6 = r4 * r2
            radial = 1.0 + self.dist_coeffs[0]*r2 + self.dist_coeffs[1]*r4 + self.dist_coeffs[4]*r6
            x_dist = x * radial + 2*self.dist_coeffs[2]*x*y + self.dist_coeffs[3]*(r2 + 2*x*x)
            y_dist = y * radial + self.dist_coeffs[2]*(r2 + 2*y*y) + 2*self.dist_coeffs[3]*x*y
            x, y = x_dist, y_dist

        px = self.K[0, 0] * x + self.K[0, 2]
        py = self.K[1, 1] * y + self.K[1, 2]

        coords = np.vstack((px, py)).T
        if return_mask:
            return coords, mask
        return coords

    def point_at(self, ra_deg: float, dec_deg: float) -> None:
        ra  = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)
        forward = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)], dtype=float)

        up_guess = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(up_guess, forward)) > 0.99:
            up_guess = np.array([0.0, 1.0, 0.0])

        right = np.cross(up_guess, forward); right /= np.linalg.norm(right)
        up    = np.cross(forward, right);    up    /= np.linalg.norm(up)
        self.R = np.column_stack((right, up, forward))
