import numpy as np
from skimage import color

def rgb2lab_normalize(image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to the LAB color space and normalizes the values to the range [-1, 1].
    
    Args:
        image (np.ndarray): The input RGB image as a NumPy array with shape (H, W, 3).

    Returns:
        np.ndarray: The LAB image with values normalized to the range [-1, 1].
    """
    lab_image = color.rgb2lab(image)  # Convert RGB to LAB
    # Normalize L, a, b to the range [-1, 1]
    lab_image[..., 0] = (lab_image[..., 0] / 50.0) - 1.0  # Normalize L: [0, 100] -> [-1, 1]
    lab_image[..., 1:] = lab_image[..., 1:] / 128.0       # Normalize a, b: [-128, 127] -> [-1, 1]
    return lab_image

def lab2rgb_denormalize(image: np.ndarray) -> np.ndarray:
    """
    Converts a normalized LAB image back to the RGB color space.
    
    Args:
        image (np.ndarray): The normalized LAB image as a NumPy array with shape (H, W, 3),
                            with values in the range [-1, 1].

    Returns:
        np.ndarray: The RGB image.
    """
    # Denormalize L, a, b to their original range
    lab_image = np.empty_like(image)
    lab_image[0, ...] = (image[0, ...] + 1.0) * 50.0   # Denormalize L: [-1, 1] -> [0, 100]
    lab_image[1:, ...] = image[1:, ...] * 128.0        # Denormalize a, b: [-1, 1] -> [-128, 127]
    lab_image = lab_image.transpose((1, 2, 0))
    rgb_image = color.lab2rgb(lab_image)  # Convert LAB back to RGB
    return rgb_image
