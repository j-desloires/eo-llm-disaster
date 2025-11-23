from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from loguru import logger


def load_raster(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loads a raster file (like a GeoTIFF) into a NumPy array and its metadata.

    Args:
        path: The file path to the raster image.

    Returns:
        A tuple containing:
        - The image data as a NumPy array (bands, height, width).
        - The raster's metadata profile (CRS, transform, etc.).
    """
    try:
        with rasterio.open(path) as dataset:
            image = dataset.read()
            profile = dataset.profile
            logger.info(
                f"Loaded raster '{path}' with shape {image.shape} and CRS '{profile['crs']}'."
            )
            return image, profile
    except rasterio.errors.RasterioIOError as e:
        logger.error(f"Could not read raster file at '{path}': {e}")
        raise


def normalize_image(
    image: np.ndarray,
    min_val: float = 0,
    max_val: float = 255,
    target_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Normalizes image pixel values to a specified target range.

    Args:
        image: The input image as a NumPy array.
        min_val: The minimum possible pixel value in the original image.
        max_val: The maximum possible pixel value in the original image.
        target_range: A tuple (min, max) for the output range.

    Returns:
        The normalized image as a NumPy array with float data type.
    """
    if image.max() > max_val or image.min() < min_val:
        logger.warning("Image values are outside the expected min/max range. Clipping.")
        image = np.clip(image, min_val, max_val)

    # Scale to [0, 1] first, then to the target range
    image_norm = (image.astype(np.float32) - min_val) / (max_val - min_val)
    target_min, target_max = target_range
    image_scaled = image_norm * (target_max - target_min) + target_min

    return image_scaled


def create_patches(
    image: np.ndarray, patch_size: int = 256, stride: int = 128
) -> List[np.ndarray]:
    """
    Splits a large image into smaller, possibly overlapping patches.

    Assumes image shape is (bands, height, width).

    Args:
        image: The input image as a NumPy array.
        patch_size: The height and width of each square patch.
        stride: The step size to move between patches. A smaller stride
                than patch_size results in overlapping patches.

    Returns:
        A list of image patches, each as a NumPy array.
    """
    if image.ndim != 3:
        raise ValueError(
            f"Input image must be 3-dimensional (bands, height, width), but got {image.ndim} dims."
        )

    _, height, width = image.shape
    patches = []

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[:, y : y + patch_size, x : x + patch_size]
            patches.append(patch)

    logger.info(
        f"Created {len(patches)} patches of size {patch_size}x{patch_size} "
        f"from image of size {height}x{width}."
    )
    return patches
