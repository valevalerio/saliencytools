"""
 This module contains the functions used to compare 
 different saliency maps.

Metrics added: 
    - ResNet Feature Similarity (exclude the last layer, compare the extracted features)
    - ShapGap Cosine
    - ShapGap L2 and 
    - Earth Mover's Distance (EMD)
    - Mean Absolute Error
    - Sign Agreement Ratio (SAR)
    - Sign Distance
    - Intersection over Union
    - Correlation Distance
    - Mean Squared Error
    - Peak Signal to Noise Ratio (PSNR)
    - Czenakowski Distance
    - Jaccard Index
    - Jaccard Distance
    - SSIM (Structural Similarity Index Measure)

"""

import numpy as np
import torch.nn.functional as F
import torch
from skimage import metrics
from scipy.stats import wasserstein_distance
# normalization functions
def normalize_mask(mask):
    """
    Normalize the mask to the range [-1, 1]
    """
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    mask = 2 * mask - 1
    return mask
def normalize_mask_0_1(mask):
    """
    Normalize the mask to the range [0, 1]
    """
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    return mask
def clip_mask(mask):
    """
    Clip the mask to the range [-1, 1]
    """
    return np.clip(mask, -1, 1)


def euclidean_distance(a, b):
    """
    Compute the Euclidean distance between two images.
    """
    return np.sqrt(np.sum((a - b) ** 2))
def cosine_distance(a, b):
    """
    Compute the cosine distance between two vectors
    """
    a = a.flatten()
    b = b.flatten()
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def emd(a, b):
    return wasserstein_distance(a.flatten(), b.flatten())
def mean_absolute_error(a, b):
    """
    Compute the mean absolute error between two images.
    """
    return np.mean(np.abs(a - b))
# Sign Agreement Ratio (SAR)
def sign_agreement_ratio(a, b):
    """
    Compute the sign agreement ratio between two images.
    """
    a = a.flatten()
    b = b.flatten()
    return 1 - np.mean(np.sign(a) == np.sign(b))
def sign_distance(a, b):
    """
    Compute the sign distance between two images.
    """
    a = a.flatten()
    b = b.flatten()
    return np.mean(np.sign(a) != np.sign(b))
def intersection_over_union(a, b):
    """
    Compute the intersection over union between two images.
    """
    a = a.flatten()
    b = b.flatten()
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))
    return 1 - intersection / union
def correlation_distance(a, b):
    """
    Compute the correlation distance between two images.
    """
    a = a.flatten()
    b = b.flatten()
    return 1 - np.corrcoef(a, b)[0, 1]
def mean_squared_error(a, b):
    """
    Compute the mean squared error between two images.
    """
    return np.mean((a - b) ** 2)
def ssim(a, b):
    """
    Compute the structural similarity index between two images.
    """
    return (1-metrics.structural_similarity(a, b, full=False,
                                  data_range= np.maximum(a.max(), b.max()) - np.minimum(a.min(), b.min())))/2

def psnr(a, b):
    """
    Compute the peak signal to noise ratio between two images.
    """
    return metrics.peak_signal_noise_ratio(a, b,
                                           data_range= np.maximum(a.max(), b.max()) - np.minimum(a.min(), b.min()))
def czenakowski_distance(a, b):
    """
    Compute the Czenakowski distance between two images.
    """
    # Calculate the sum of minimums (intersection)
    sum_of_minimums = np.sum(np.minimum(a, b))
    
    # Calculate the sum of the values
    sum_of_values = np.sum(a + b)
    
    # Calculate Czekanowski distance
    # Formula: 1 - (2 * sum of minimums) / (sum of all values)
    if sum_of_values == 0:
        return 0  # If both images are all zeros, they're identical
    return 1 - (2 * sum_of_minimums) / sum_of_values
def jaccard_index(a, b):
    """
    Compute the Jaccard index between two images.
    """
    # Flatten the images
    a = a.flatten()
    b = b.flatten()
    
    # Calculate intersection and union
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))
    
    # Jaccard index formula
    if union == 0:
        return 0  # If both images are all zeros, they're identical
    return intersection / union
def jaccard_distance(a, b):
    """
    Compute the Jaccard distance between two images.
    (uses the Jaccard index)
    """
    return 1 - jaccard_index(a, b)




# specify the names of the metrics

# in the __init__.py file we import the functions and classes we want to expose as this:
# from .maskcompare import (
#     normalize_mask,
#     clip_mask,
#     cosine_distance,
#     emd,
#     correlation_distance,
#     czenakowski_distance,
#     l2_distance,
#     mean_absolute_error,
#     sign_agreement_ratio,
#     sign_distance,
#     intersection_over_union,
#     jaccard_index,
#     jaccard_distance,
#     ssim,
#     psnr,
#     mean_squared_error,
#     )
# and we add the __name__ attribute to the functions to make them more readable

cosine_distance.__name__ = "$ShapGap_{Cosine}$"
euclidean_distance.__name__ = "$ShapGap_{L2}$"
emd.__name__ = "Earth Mover's Distance"
mean_absolute_error.__name__ = "MAE"
sign_agreement_ratio.__name__ = "Sign Agreem Ratio"
sign_distance.__name__ = "Sign Distance"
intersection_over_union.__name__ = "Inter over Union"
correlation_distance.__name__ = "Correlation Distance"
mean_squared_error.__name__ = "MSE"
ssim.__name__ = "Structural Similarity"
psnr.__name__ = "Peak Signal Noise Ratio"
czenakowski_distance.__name__ = "Czekanowski Distance"
jaccard_distance.__name__ = "Jaccard Distance"
jaccard_index.__name__ = "Jaccard Index"









