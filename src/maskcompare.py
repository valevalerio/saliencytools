"""
This module contains functions to compare different saliency maps using various metrics.

Metrics implemented:
    - ResNet Feature Similarity (exclude the last layer, compare the extracted features)
    - ShapGap Cosine
    - ShapGap L2
    - Earth Mover's Distance (EMD)
    - Mean Absolute Error (MAE)
    - Sign Agreement Ratio (SAR)
    - Sign Distance
    - Intersection over Union (IoU)
    - Correlation Distance
    - Mean Squared Error (MSE)
    - Peak Signal-to-Noise Ratio (PSNR)
    - Czekanowski Distance
    - Jaccard Index
    - Jaccard Distance
    - Structural Similarity Index Measure (SSIM)

References:

"""

import numpy as np
import torch.nn.functional as F
import torch
from skimage import metrics
from scipy.stats import wasserstein_distance

# Normalization functions
def normalize_mask(mask):
    """
    Normalize the mask to the range [-1, 1].

    Parameters:
        mask (numpy.ndarray): Input saliency map.

    Returns:
        numpy.ndarray: Normalized saliency map.
    """
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    mask = 2 * mask - 1
    return mask

def normalize_mask_0_1(mask):
    """
    Normalize the mask to the range [0, 1].

    Parameters:
        mask (numpy.ndarray): Input saliency map.

    Returns:
        numpy.ndarray: Normalized saliency map.
    """
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    return mask

def clip_mask(mask):
    """
    Clip the mask to the range [-1, 1].

    Parameters:
        mask (numpy.ndarray): Input saliency map.

    Returns:
        numpy.ndarray: Clipped saliency map.
    """
    return np.clip(mask, -1, 1)

# Distance metrics
def euclidean_distance(a, b):
    """
    Compute the Euclidean distance between two images.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Euclidean distance.
    """
    return np.sqrt(np.sum((a - b) ** 2))

def cosine_distance(a, b):
    """
    Compute the cosine distance between two vectors.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Cosine distance.
    """
    a = a.flatten()
    b = b.flatten()
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def emd(a, b):
    """
    Compute the Earth Mover's Distance (EMD) between two images.

    Reference:
        Rubner, Y., Tomasi, C., & Guibas, L. J. (2000). 
        "The earth mover's distance as a metric for image retrieval."

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Earth Mover's Distance.
    """
    return wasserstein_distance(a.flatten(), b.flatten())

def mean_absolute_error(a, b):
    """
    Compute the Mean Absolute Error (MAE) between two images.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Mean Absolute Error.
    """
    return np.mean(np.abs(a - b))

def sign_agreement_ratio(a, b):
    """
    Compute the Sign Agreement Ratio (SAR) between two images.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Sign Agreement Ratio.
    """
    a = a.flatten()
    b = b.flatten()
    return 1 - np.mean(np.sign(a) == np.sign(b))

def sign_distance(a, b):
    """
    Compute the Sign Distance between two images.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Sign Distance.
    """
    a = a.flatten()
    b = b.flatten()
    return np.mean(np.sign(a) != np.sign(b))

def intersection_over_union(a, b):
    """
    Compute the Intersection over Union (IoU) between two images.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Intersection over Union.
    """
    a = a.flatten()
    b = b.flatten()
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))
    return 1 - intersection / union

def correlation_distance(a, b):
    """
    Compute the Correlation Distance between two images.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Correlation Distance.
    """
    a = a.flatten()
    b = b.flatten()
    return 1 - np.corrcoef(a, b)[0, 1]

def mean_squared_error(a, b):
    """
    Compute the Mean Squared Error (MSE) between two images.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Mean Squared Error.
    """
    return np.mean((a - b) ** 2)

def ssim(a, b):
    """
    Compute the Structural Similarity Index Measure (SSIM) between two images.

    Reference:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). 
        "Image quality assessment: From error visibility to structural similarity."

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: SSIM value.
    """
    return (1 - metrics.structural_similarity(a, b, full=False,
                                              data_range=np.maximum(a.max(), b.max()) - np.minimum(a.min(), b.min()))) / 2

def psnr(a, b):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Reference:
        Huynh-Thu, Q., & Ghanbari, M. (2008). 
        "Scope of validity of PSNR in image/video quality assessment."

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: PSNR value.
    """
    return metrics.peak_signal_noise_ratio(a, b,
                                           data_range=np.maximum(a.max(), b.max()) - np.minimum(a.min(), b.min()))

def czenakowski_distance(a, b):
    """
    Compute the Czekanowski Distance between two images.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Czekanowski Distance.
    """
    sum_of_minimums = np.sum(np.minimum(a, b))
    sum_of_values = np.sum(a + b)
    if sum_of_values == 0:
        return 0  # If both images are all zeros, they're identical
    return 1 - (2 * sum_of_minimums) / sum_of_values

def jaccard_index(a, b):
    """
    Compute the Jaccard Index between two images.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Jaccard Index.
    """
    a = a.flatten()
    b = b.flatten()
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))
    if union == 0:
        return 0  # If both images are all zeros, they're identical
    return intersection / union

def jaccard_distance(a, b):
    """
    Compute the Jaccard Distance between two images.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Jaccard Distance.
    """
    return 1 - jaccard_index(a, b)

# Assign readable names to metrics
cosine_distance.__name__ = "$ShapGap_{Cosine}$"
euclidean_distance.__name__ = "$ShapGap_{L2}$"
emd.__name__ = "Earth Mover's Distance"
mean_absolute_error.__name__ = "MAE"
sign_agreement_ratio.__name__ = "Sign Agreement Ratio"
sign_distance.__name__ = "Sign Distance"
intersection_over_union.__name__ = "Intersection over Union"
correlation_distance.__name__ = "Correlation Distance"
mean_squared_error.__name__ = "MSE"
ssim.__name__ = "SSIM"
psnr.__name__ = "PSNR"
czenakowski_distance.__name__ = "Czekanowski Distance"
jaccard_distance.__name__ = "Jaccard Distance"
jaccard_index.__name__ = "Jaccard Index"
