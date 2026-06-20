import numpy as np
import cv2
from typing import Optional
from dataclasses import dataclass

@dataclass
class NoiseParams:
    """Parameters for different types of noise and optical effects."""
    # Gaussian noise
    gaussian_mean: float = 0.0
    gaussian_sigma: float = 0.0
    
    # Hot pixels
    hot_pixel_prob: float = 0.0
    hot_pixel_intensity: int = 255
    
    # Cosmic rays
    cosmic_ray_prob: float = 0.0
    cosmic_ray_min_length: int = 3
    cosmic_ray_max_length: int = 20
    cosmic_ray_intensity: int = 200
    
    # Vignetting
    vignette_strength: float = 0.0  # 0 = no vignetting, 1 = full vignette
    
    # Chromatic aberration
    chromatic_aberration: float = 0.0  # 0 = none, >0 = stronger effect
    
    # Bloom effect
    bloom_threshold: float = 0.8  # 0-1, higher = only bright stars bloom
    bloom_intensity: float = 0.0  # 0 = no bloom, >0 = stronger bloom
    
    # Star twinkling
    twinkle_amount: float = 0.0  # 0 = no twinkling, >0 = stronger twinkling
    twinkle_speed: float = 1.0   # Speed of twinkling effect

def add_gaussian_noise(image: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Add Gaussian noise to the image."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_hot_pixels(image: np.ndarray, prob: float, intensity: int) -> np.ndarray:
    """Add hot pixels to the image."""
    if prob <= 0 or intensity <= 0:
        return image
        
    mask = np.random.random(image.shape) < prob
    result = image.copy()
    result[mask] = np.maximum(result[mask], intensity)
    return result

def add_cosmic_rays(image: np.ndarray, prob: float, min_length: int, 
                   max_length: int, intensity: int) -> np.ndarray:
    """Add cosmic ray artifacts to the image."""
    if prob <= 0 or intensity <= 0:
        return image
        
    result = image.copy()
    h, w = image.shape
    
    # Number of cosmic rays is based on image area and probability
    num_rays = int(prob * h * w / 1000)
    
    for _ in range(num_rays):
        # Random start point
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        length = np.random.randint(min_length, max_length)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Draw line segment
        x2 = int(x + length * np.cos(angle))
        y2 = int(y + length * np.sin(angle))
        
        # Clip to image bounds
        x1, y1 = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
        x2, y2 = np.clip(x2, 0, w-1), np.clip(y2, 0, h-1)
        
        # Draw the cosmic ray
        cv2.line(result, (x1, y1), (x2, y2), intensity, 1, lineType=cv2.LINE_AA)
        
    return result

def apply_vignette(image: np.ndarray, strength: float) -> np.ndarray:
    """Apply vignette effect to the image."""
    if strength <= 0:
        return image
        
    h, w = image.shape[:2]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    
    # Create vignette mask (1 in center, 0 at edges)
    radius = np.sqrt(X**2 + Y**2) / np.sqrt(2)  # Normalize to [0, 1]
    mask = 1 - strength * radius
    mask = np.clip(mask, 0, 1)
    
    # Apply mask to image
    result = (image.astype(np.float32) * mask[..., np.newaxis] if len(image.shape) == 3 
             else image.astype(np.float32) * mask)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_chromatic_aberration(image: np.ndarray, amount: float) -> np.ndarray:
    """Apply chromatic aberration effect to the image."""
    if amount <= 0 or len(image.shape) != 3:
        return image
        
    # Split into color channels
    b, g, r = cv2.split(image)
    
    # Shift red and blue channels
    shift = int(2 * amount)
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    shifted_r = cv2.warpAffine(r, M, (r.shape[1], r.shape[0]))
    
    M = np.float32([[1, 0, -shift], [0, 1, 0]])
    shifted_b = cv2.warpAffine(b, M, (b.shape[1], b.shape[0]))
    
    # Merge channels back
    return cv2.merge([shifted_b, g, shifted_r])

def apply_bloom_effect(image: np.ndarray, threshold: float, intensity: float) -> np.ndarray:
    """Apply a bloom effect to bright areas of the image."""
    if intensity <= 0 or threshold <= 0:
        return image
        
    # Create a mask of bright areas
    _, bright_mask = cv2.threshold(image, threshold * 255, 255, cv2.THRESH_BINARY)
    bright_mask = bright_mask.astype(np.uint8)
    
    # Blur the bright areas
    bloom = cv2.GaussianBlur(bright_mask, (0, 0), 5)
    
    # Add the bloom effect to the original image
    result = cv2.addWeighted(image, 1.0, bloom, intensity, 0)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_star_twinkling(image: np.ndarray, stars: np.ndarray, 
                        time: float, amount: float, speed: float) -> np.ndarray:
    """Apply a twinkling effect to stars."""
    if amount <= 0 or len(stars) == 0:
        return image
        
    result = image.copy()
    
    for (x, y, mag, size) in stars:
        # Calculate twinkle factor based on time and position
        twinkle = np.sin(time * speed + x * 0.1 + y * 0.1) * amount + 1.0
        
        # Get the current brightness
        brightness = image[int(y), int(x)]
        
        # Apply twinkle
        new_brightness = np.clip(brightness * twinkle, 0, 255).astype(np.uint8)
        
        # Update the star
        cv2.circle(result, (int(x), int(y)), int(size), int(new_brightness), -1)
    
    return result

def render_starfield(camera, catalog, noise_params: Optional[NoiseParams] = None, 
                    time: float = 0.0) -> np.ndarray:
    """
    Render a starfield with optional noise and optical effects.
    
    Args:
        camera: Camera object with projection parameters
        catalog: Star catalog with positions and magnitudes
        noise_params: Parameters for noise and optical effects
        time: Current time for time-based effects
        
    Returns:
        Rendered image with stars and applied effects
    """
    if noise_params is None:
        noise_params = NoiseParams()
    
    h, w = camera.height, camera.width

    # Star directions and magnitudes
    if all(c in catalog.columns for c in ('x', 'y', 'z')):
        dirs = catalog[['x', 'y', 'z']].values.astype(np.float64)
    elif all(c in catalog.columns for c in ('ux', 'uy', 'uz')):
        dirs = catalog[['ux', 'uy', 'uz']].values.astype(np.float64)
    else:
        raise RuntimeError('Catalog missing unit-vector columns (x,y,z or ux,uy,uz)')

    mags = catalog['mag'].values.astype(np.float64)
    
    # Project stars to image coordinates
    coords, mask = camera.project(dirs, return_mask=True, apply_distortion=True)
    if coords.size == 0 or not np.any(mask):
        return np.zeros((camera.height, camera.width), dtype=np.uint8)
    
    stars = []
    drawn = 0

    scale = 2
    img_highres = np.zeros((h * scale, w * scale), dtype=np.float32)
    
    for (x, y), mag in zip(coords, mags[mask]):
        if 0 <= x < w and 0 <= y < h:
            # Calculate brightness based on magnitude
            rel = pow(10.0, -0.4 * (mag - 2.0))
            brightness = np.clip(np.sqrt(rel) * 255.0, 80, 255)
            size = max(1, int(5 - mag * 0.5))  # Brighter stars are larger
            
            # Store star info for effects
            stars.append((x, y, mag, size))
            
            # Draw star at higher resolution
            x_hr, y_hr = int(round(x * scale)), int(round(y * scale))
            size_hr = max(1, size * scale // 2)
            cv2.circle(img_highres, (x_hr, y_hr), size_hr, brightness, -1)
            drawn += 1
    
    # If no stars were drawn but we have coordinates, draw at least one
    if drawn == 0 and coords.size > 0:
        cx, cy = w / 2.0, h / 2.0
        d2 = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
        idx = int(np.argmin(d2))
        x, y = coords[idx]
        mag = float(mags[mask][idx])
        rel = pow(10.0, -0.4 * (mag - 2.0))
        brightness = np.clip(np.sqrt(rel) * 255.0, 120, 255)
        size = 5
        
        x_hr, y_hr = int(round(x * scale)), int(round(y * scale))
        size_hr = size * scale // 2
        cv2.circle(img_highres, (x_hr, y_hr), size_hr, brightness, -1)
        stars.append((x, y, mag, size))
    
    # Downsample the high-res image to get anti-aliased stars
    img = cv2.resize(img_highres, (w, h), interpolation=cv2.INTER_AREA).astype(np.uint8)
    
    # Apply Gaussian blur for natural star appearance
    if img.max() > 0:
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
    
    if noise_params.twinkle_amount > 0 and len(stars) > 0:
        img = apply_star_twinkling(img, stars, time,
                                   noise_params.twinkle_amount,
                                   noise_params.twinkle_speed)

    # Apply bloom effect to bright stars
    if noise_params.bloom_intensity > 0:
        img = apply_bloom_effect(img, noise_params.bloom_threshold, 
                               noise_params.bloom_intensity)
    
    # Add Gaussian noise
    if noise_params.gaussian_sigma > 0:
        img = add_gaussian_noise(img, noise_params.gaussian_mean, 
                               noise_params.gaussian_sigma)
    
    # Add hot pixels
    if noise_params.hot_pixel_prob > 0:
        img = add_hot_pixels(img, noise_params.hot_pixel_prob, 
                           noise_params.hot_pixel_intensity)
    
    # Add cosmic rays
    if noise_params.cosmic_ray_prob > 0:
        img = add_cosmic_rays(img, noise_params.cosmic_ray_prob, 
                            noise_params.cosmic_ray_min_length,
                            noise_params.cosmic_ray_max_length,
                            noise_params.cosmic_ray_intensity)
    
    # Apply vignette effect
    if noise_params.vignette_strength > 0:
        img = apply_vignette(img, noise_params.vignette_strength)
    
    # Apply chromatic aberration (convert to color first)
    if noise_params.chromatic_aberration > 0:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_color = apply_chromatic_aberration(img_color, noise_params.chromatic_aberration)
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    return img