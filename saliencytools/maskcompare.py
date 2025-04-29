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
"""

import numpy as np
import torch.nn.functional as F
import torch
from skimage import metrics
from scipy.stats import wasserstein_distance
from scipy.ndimage import sobel

# ============= Normalization Functions =============
def make_histogram(mask: np.ndarray, bins: int = 256) -> np.ndarray:
    """Convert continuous values to discrete distribution
    This function takes a saliency map and converts it into a histogram representation.
    The histogram is normalized to ensure that the sum of all bins equals 1,
    making it suitable for comparing distributions.
    Parameters:
        mask (numpy.ndarray): Input saliency map. This is a 2D or 3D array 
                              representing the saliency values of an image.
        bins (int): Number of bins for the histogram. Default is 256.
    Returns:
        numpy.ndarray: Normalized histogram of the saliency map. The sum of all 
                       bins equals 1, representing the distribution of saliency values.
    """
    hist, _ = np.histogram(mask, bins=bins, density=True)
    return hist / (np.sum(hist) + 1e-8)

def normalize_mask(mask):
    """
    Normalize the mask to the range [-1, 1].

    This function rescales the input saliency map to the range [-1, 1], 
    ensuring that the values are standardized for further processing. 
    This normalization is particularly useful when working with metrics 
    or models that expect inputs in this range.

    Parameters:
        mask (numpy.ndarray): Input saliency map. This is a 2D or 3D array 
                              representing the saliency values of an image.

    Returns:
        numpy.ndarray: Normalized saliency map with values in the range [-1, 1].
    """
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    mask = 2 * mask - 1
    return mask

def normalize_mask_0_1(mask):
    """
    Normalize the input saliency map to the range [0, 1].

    This function ensures that the values in the input saliency map are scaled 
    to lie within the range [0, 1]. This is useful for standardizing the input 
    data for further processing or comparison, especially when working with 
    metrics that require normalized inputs.

    Parameters:
        mask (numpy.ndarray): Input saliency map. This is a 2D or 3D array 
                              representing the saliency values of an image, 
                              where higher values indicate greater importance.

    Returns:
        numpy.ndarray: Normalized saliency map with values in the range [0, 1]. 
                       The output has the same shape as the input.
    """
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)
    return mask

def clip_mask(mask):
    """
    Clip the mask to the range [-1, 1].

    This function ensures that the values in the input saliency map do not 
    exceed the range [-1, 1]. This is useful for preventing outliers or 
    extreme values from affecting downstream computations.

    Parameters:
        mask (numpy.ndarray): Input saliency map. This is a 2D or 3D array 
                              representing the saliency values of an image.

    Returns:
        numpy.ndarray: Clipped saliency map with values constrained to [-1, 1].
    """
    return np.clip(mask, -1, 1)

# ============= Geometric Distances =============
def euclidean_distance(a, b):
    """
    Compute the Euclidean distance between two images.

    The Euclidean distance measures the straight-line distance between 
    corresponding pixels in two images. It captures the overall magnitude 
    of differences between the two images.

    Reference:
        Commonly used in image processing and computer vision literature, also known as Frobenius norm.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Euclidean distance, representing the magnitude of differences.
    """
    return np.sqrt(np.sum((a - b) ** 2))

def cosine_distance(a, b):
    """
    Compute the cosine distance between two vectors.

    The cosine distance measures the angular difference between two vectors 
    in a high-dimensional space. It is useful for comparing the orientation 
    of two saliency maps rather than their magnitude.

    Reference:
        Commonly used in vector similarity and machine learning literature.
        

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Cosine distance, representing the angular difference.
    """
    a = a.flatten()
    b = b.flatten()
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def mean_absolute_error(a, b):
    """
    Compute the Mean Absolute Error (MAE) between two images.

    The MAE measures the average absolute difference between corresponding 
    pixels in two images. It captures the overall deviation in pixel values.

    Reference:
        Commonly used in regression analysis and image processing.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Mean Absolute Error, representing the average deviation.
    """
    return np.mean(np.abs(a - b))

def mean_squared_error(a, b):
    """
    Compute the Mean Squared Error (MSE) between two images.

    The MSE measures the average squared difference between corresponding 
    pixels in two images. It emphasizes larger deviations more than the 
    Mean Absolute Error.

    Reference:
        Commonly used in regression analysis and image processing.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Mean Squared Error, representing the average squared deviation.
    """
    return np.mean((a - b) ** 2)

# ============= Distribution/Statistical Distances =============
def emd(a, b,bins=256):
    """
    Compute the Earth Mover's Distance (EMD) between two images.

    The EMD measures the minimum cost of transforming one distribution 
    into another. It is particularly useful for comparing saliency maps 
    with spatial distributions of importance.

    Reference:
        - Rubner, Y., Tomasi, C., & Guibas, L. J. (2000). The Earth Mover's Distance as a Metric for Image Retrieval. *International Journal of Computer Vision*, 40(2), 99–121. https://doi.org/10.1023/A:1026543900054

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Earth Mover's Distance, representing the cost of transformation.
    """
    a_hist = make_histogram(a, bins)
    b_hist = make_histogram(b, bins)
    return wasserstein_distance(np.arange(bins), np.arange(bins), a_hist, b_hist)

def correlation_distance(a, b):
    """
    Compute the Correlation Distance between two images.

    The Correlation Distance measures the linear relationship between 
    corresponding pixel values in two images. It captures how well the 
    variations in one image are correlated with the other.

    Reference:
        Commonly used in statistics and signal processing.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Correlation Distance, representing the inverse of correlation.
    """
    a = a.flatten()
    b = b.flatten()
    return 1 - np.corrcoef(a, b)[0, 1]

def psnr(a, b):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    The PSNR measures the ratio between the maximum possible pixel value 
    and the mean squared error. It is commonly used to evaluate the quality 
    of reconstructed images.

    Reference:
        Huynh-Thu, Q., & Ghanbari, M. (2008). 
        "Scope of validity of PSNR in image/video quality assessment."

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: PSNR value, representing the signal-to-noise ratio.
    """
    return 1/metrics.peak_signal_noise_ratio(a, b,
                                           data_range=np.maximum(a.max(), b.max()) - np.minimum(a.min(), b.min()))

# ============= Set-Theoretic Distances ==========
def jaccard_index(a, b):
    """
    Compute the Jaccard Index between two images.

    The Jaccard Index measures the similarity between two images by comparing 
    the intersection and union of their pixel values. It is commonly used for 
    evaluating binary or thresholded saliency maps.

    Reference:
        Commonly used in set theory and image segmentation literature.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Jaccard Index, representing the similarity ratio.
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

    The Jaccard Distance is the complement of the Jaccard Index and measures 
    the dissimilarity between two images. It is useful for evaluating the 
    differences between binary or thresholded saliency maps.

    Reference:
        Commonly used in set theory and image segmentation literature.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Jaccard Distance, representing the dissimilarity ratio.
    """
    return 1 - jaccard_index(a, b)

def czenakowski_distance(a, b):
    """
    Compute the Czekanowski Distance between two images.

    The Czekanowski Distance measures the dissimilarity between two images 
    based on the ratio of their minimum and total pixel values. It is useful 
    for comparing distributions with overlapping regions.

    Reference:
        T. SORENSEN (1948) "A method of establishing groups of equal amplitude in plant sociology based on similarity of species content and its application to analyses of the vegetation on danish commons." Biologiske Skrifter.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Czekanowski Distance, representing the dissimilarity.
    """
    sum_of_minimums = np.sum(np.minimum(a, b))
    sum_of_values = np.sum(a + b)
    if sum_of_values == 0:
        return 0  # If both images are all zeros, they're identical
    return 1 - (2 * sum_of_minimums) / sum_of_values

# ============= Binary Distances =================
def sign_agreement_ratio(a, b):
    """
    Compute the Sign Agreement Ratio (SAR) between two images.

    The SAR measures the proportion of pixels where the signs of the values 
    in two images agree. It captures the consistency in the direction of 
    importance between two saliency maps.

    Reference:
        A. M. Nevill, G. Atkinson (1997) "Assessing agreement between measurements recorded on a ratio scale" in sports medicine and sports science



    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Sign Agreement Ratio, representing the proportion of agreement.
    """
    a = a.flatten()
    b = b.flatten()
    return 1 - np.mean(np.sign(a) == np.sign(b))

# ============= Structural Distances =============
def ssim(a, b):
    """
    Compute the Structural Similarity Index Measure (SSIM) between two images.

    The SSIM evaluates the perceptual similarity between two images by 
    considering luminance, contrast, and structure. It is widely used for 
    assessing image quality and similarity.

    Reference:
        - Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861
    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: SSIM value, representing the perceptual similarity.
    """
    if np.allclose(a, b):
        return 0
    return (1 - metrics.structural_similarity(a, b, full=False,
                                              data_range=np.maximum(a.max(), b.max()) - np.minimum(a.min(), b.min()))) / 2

# ============= Pixel-wise Distances ============= 
# These functions return an distance matrix so that we can visualize the distance between each pixel in the two images
# abs_error_lambda = lambda a, b: np.abs(a - b)
# squared_error_lambda = lambda a, b: (a - b) ** 2
# minkowski_lambda = lambda a, b: np.linalg.norm(a - b, ord=2)




# Assign readable names to metrics
# ------------- Geometric metrics
euclidean_distance.__name__ = "$ShapGap_{L2}$"
cosine_distance.__name__ = "$ShapGap_{Cosine}$"
mean_absolute_error.__name__ = "MAE"
mean_squared_error.__name__ = "MSE"

# ------------- Distribution/Statistical metrics
emd.__name__ = "Earth Mover's Distance"
correlation_distance.__name__ = "Correlation Distance"
psnr.__name__ = "PSNR"

# ------------- Set Theory metrics
jaccard_index.__name__ = "Jaccard Index"
jaccard_distance.__name__ = "Jaccard Distance"
czenakowski_distance.__name__ = "Czekanowski Distance"

# ------------- Binary metrics
sign_agreement_ratio.__name__ = "Sign Agreement Ratio"

# ------------- Structural metrics
ssim.__name__ = "SSIM"
