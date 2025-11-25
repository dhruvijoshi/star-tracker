import numpy as np
from scipy.spatial.transform import Rotation as R

class Camera:
    def __init__(self, width=640, height=480, fov_deg=60):
        self.width = int(width)
        self.height = int(height)
        self.fov = np.deg2rad(fov_deg)
        
        # Compute focal length from FOV
        self.focal = (self.width / 2.0) / np.tan(self.fov / 2.0)
        self.R = np.eye(3)

    def set_orientation_random(self):
        # Assign a random pointing direction
        self.R = R.random().as_matrix()
    
    def project(self, dirs, return_mask=False):
        # Camera axes in world coordinates (columns of R)
        rx, ry, rz = self.R[:, 0], self.R[:, 1], self.R[:, 2]
        # Camera-space coordinates are dot products with axes
        x = dirs @ rx
        y = dirs @ ry
        z = dirs @ rz
        
        # Keep stars in front of the camera (+Z in camera space)
        mask = z > 1e-8
        if not np.any(mask):
            if return_mask:
                return np.zeros((0, 2), dtype=np.float32), mask
            return np.zeros((0, 2), dtype=np.float32)
        
        inv_z = 1.0 / z[mask]

        # Perspective projection with conventional image Y-axis pointing down
        px = self.focal * (x[mask] * inv_z) + (self.width / 2.0)
        py = (self.height / 2.0) - self.focal * (y[mask] * inv_z)
        
        coords = np.vstack((px, py)).T
        if return_mask:
            return coords, mask
        return coords
        
    def point_at(self, ra_deg, dec_deg):
        # Convert RA/Dec degrees to radians
        ra = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)

        # Convert RA/Dec to unit direction (target look direction)
        tx = np.cos(dec) * np.cos(ra)
        ty = np.cos(dec) * np.sin(ra)
        tz = np.sin(dec)
        forward = np.array([tx, ty, tz], dtype=float)

        # Choose a stable global up guess not parallel to forward
        up_guess = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(up_guess, forward)) > 0.99:
            up_guess = np.array([0.0, 1.0, 0.0], dtype=float)

        # Build a right-handed camera basis: +Z = forward, +X = right, +Y = up
        right = np.cross(up_guess, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        self.R = np.column_stack((right, up, forward))
