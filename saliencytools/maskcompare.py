"""
This module contains functions to compare different saliency maps using various metrics.

Metrics implemented:
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
    - KL Divergence (symmetric / Jeffrey divergence)
    - AUC-Judd (symmetric pairwise variant)

"""

import numpy as np
from skimage import metrics
from scipy.stats import wasserstein_distance
from scipy.ndimage import sobel
from sklearn import metrics as sklearn_metrics

# ============= Normalization Functions =============
def make_histogram(mask: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Convert continuous values to discrete distribution.

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
    mask = mask / (np.max(mask) - np.min(mask) + 1e-8)
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
    mask = mask / (np.max(mask) - np.min(mask) + 1e-8)
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
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if (norm_a != norm_b) else 0.0
    return 1 - np.dot(a, b) / (norm_a * norm_b)

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
        - Rubner, Y., Tomasi, C., & Guibas, L. J. (2000). The Earth Mover's Distance as a Metric for Image Retrieval. *International Journal of Computer Vision*, 40(2), 99-121. https://doi.org/10.1023/A:1026543900054

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
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0
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

def kl_divergence(prediction, reference):
    """
    Compute the Kullback-Leibler (KL) divergence between two saliency maps.

    Reference:
        Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. *The Annals of Mathematical Statistics*, 22(1), 79-86.

    Parameters:
        prediction (numpy.ndarray): Prediction saliency map.
        reference (numpy.ndarray): Ground truth saliency map.

    Returns:
        float: KL divergence value.
    """
    prediction = prediction.flatten()
    reference = reference.flatten()

    prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction) + 1e-8)
    reference = (reference - np.min(reference)) / (np.max(reference) - np.min(reference) + 1e-8)

    prediction = prediction / (np.sum(prediction) + 1e-8)
    reference = reference / (np.sum(reference) + 1e-8)

    eps = 1e-12
    prediction = np.clip(prediction, eps, 1.0)
    reference = np.clip(reference, eps, 1.0)

    return np.sum(reference * np.log(reference / prediction))

def information_gain(a, b, baseline=None):
    """
    Compute the Information Gain (IG) between a saliency map and ground truth.

    Reference:
        Kümmerer, M., Wallis, T. S., & Bethge, M. (2015). Information-theoretic framework to overcome the ambiguity of saliency metrics. *arXiv preprint arXiv:1509.01556*.

    Parameters:
        a (numpy.ndarray): Prediction saliency map.
        b (numpy.ndarray): Ground truth saliency map.
        baseline (numpy.ndarray, optional): Baseline saliency map. Defaults to uniform.

    Returns:
        float: Information Gain value.
    """
    a = a.flatten()
    b = b.flatten()
    
    a = (a - np.min(a)) / (np.max(a) - np.min(a) + 1e-8)
    a = a / (np.sum(a) + 1e-8)
    
    if baseline is None:
        baseline = np.ones_like(a) / len(a)
    else:
        baseline = baseline.flatten()
        baseline = (baseline - np.min(baseline)) / (np.max(baseline) - np.min(baseline) + 1e-8)
        baseline = baseline / (np.sum(baseline) + 1e-8)
    
    eps = 1e-12
    a = np.clip(a, eps, 1.0)
    baseline = np.clip(baseline, eps, 1.0)
    
    return np.sum(b * (np.log2(a) - np.log2(baseline))) / (np.sum(b) + 1e-8)

def nss(a, b):
    """
    Compute the Normalized Scanpath Saliency (NSS).

    Reference:
        Peters, R. J., Iyer, A., Itti, L., & Koch, C. (2005). Components of bottom-up gaze allocation in natural scenes. *Vision Research*, 45(18), 2397-2416.

    Parameters:
        a (numpy.ndarray): Prediction saliency map.
        b (numpy.ndarray): Ground truth saliency map (fixations).

    Returns:
        float: NSS value.
    """
    if np.std(a) == 0:
        return 0.0
    a_norm = (a - np.mean(a)) / np.std(a)
    return np.sum(a_norm * b) / (np.sum(b) + 1e-8)

def linear_correlation_coefficient(a, b):
    """
    Compute the Linear Correlation Coefficient (CC).

    Parameters:
        a (numpy.ndarray): First saliency map.
        b (numpy.ndarray): Second saliency map.

    Returns:
        float: CC value.
    """
    a = a.flatten()
    b = b.flatten()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]

def auc_judd(a, b):
    """
    Compute the Area Under ROC Curve (AUC) using Judd's implementation approach.

    Reference:
        Judd, T., Ehinger, K., Durand, F., & Torralba, A. (2009). Learning to predict where humans look. *IEEE International Conference on Computer Vision (ICCV)*.

    Parameters:
        a (numpy.ndarray): Prediction saliency map.
        b (numpy.ndarray): Ground truth saliency map.

    Returns:
        float: AUC value.
    """
    a = a.flatten()
    b = b.flatten()
    
    if len(np.unique(b)) > 2:
        # Threshold ground truth if not binary
        b_bin = (b >= np.percentile(b, 90)).astype(int)
    else:
        b_bin = b.astype(int)
    
    if np.sum(b_bin) == 0 or np.sum(b_bin) == len(b_bin):
        return 0.5
        
    fpr, tpr, _ = sklearn_metrics.roc_curve(b_bin, a)
    return sklearn_metrics.auc(fpr, tpr)

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
        - Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600-612. https://doi.org/10.1109/TIP.2003.819861
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

# ============= Information-Theoretic Distances =============

def kl_divergence(a, b):
    """
    Compute the symmetric KL divergence (Jeffrey divergence) between two images.

    The standard KL(P || Q^D) from Bylinskii et al. (2017) is asymmetric:
    it measures how well a saliency prediction P approximates a ground-truth
    fixation map Q^D.  Since both maps here are peers (neither is ground truth),
    we use the symmetric variant KL(a||b) + KL(b||a), which penalises
    false positives and false negatives equally.

    Both maps are shifted to be non-negative and normalised to sum to 1
    (probability distributions) before computation.  A small epsilon
    (1e-10) is added for numerical stability, following the regularisation
    strategy of the MIT Saliency Benchmark.

    Reference:
        Bylinskii et al. (2017). "What do different evaluation metrics
        tell us about saliency models?" IEEE TPAMI, arXiv:1604.03605.

    Parameters:
        a (numpy.ndarray): First image.
        b (numpy.ndarray): Second image.

    Returns:
        float: Symmetric KL divergence (>= 0; 0 iff a == b after normalisation).
    """
    eps = 1e-10
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)

    # Shift to non-negative, add epsilon, normalise to probability distributions
    a = a - a.min() + eps
    b = b - b.min() + eps
    a /= a.sum()
    b /= b.sum()

    kl_ab = np.sum(a * np.log(a / b))
    kl_ba = np.sum(b * np.log(b / a))
    return float(kl_ab + kl_ba)


def _auc_judd_one_direction(saliency, fixations_binary):
    """
    Compute AUC-Judd for one direction (saliency as classifier of fixations).

    Uses the Wilcoxon–Mann–Whitney statistic for O(n log n) efficiency.
    """
    sal = saliency.flatten()
    fix = fixations_binary.flatten().astype(bool)
    sal_fix = sal[fix]
    sal_nonfix = sal[~fix]
    n_fix = len(sal_fix)
    n_nonfix = len(sal_nonfix)
    if n_fix == 0 or n_nonfix == 0:
        return 0.5
    sorted_nonfix = np.sort(sal_nonfix)
    # For each fixated pixel, count non-fixated pixels with strictly lower sal
    ranks = np.searchsorted(sorted_nonfix, sal_fix, side='right')
    return float(ranks.mean() / n_nonfix)


def auc_judd(prediction, reference):
    """
    Compute the AUC-Judd distance between a predicted saliency map and a
    reference map.

    AUC-Judd (Judd et al., 2009) evaluates how well a predicted saliency map
    recovers the salient regions of a reference map.  The reference is
    binarised at its mean to produce a fixation mask; the prediction is then
    treated as a continuous classifier of those fixated pixels, and the Area
    Under the ROC Curve (AUC) is reported.

    **Convention**: ``auc_judd(prediction, reference)`` — the *second* argument
    always provides the fixation mask.  This matches the intended use case:
    ``auc_judd(lime_map, shap_map)`` measures how well the LIME explanation
    recovers the regions SHAP considers important, not the reverse.

    **Asymmetry**: ``auc_judd(a, b) != auc_judd(b, a)`` in general.
    The metric therefore does not satisfy the symmetry axiom of a metric space.
    It is listed alongside SSIM and PSNR as a documented exception in the
    formal validation suite.

    In the proxy benchmark, ``metric_fn(test_image, prototype)`` places the
    test image in the prediction role and the prototype in the reference role,
    which is the natural direction: the prototype is the trusted reference, the
    test image is the explanation being evaluated.

    Reference:
        Judd et al. (2009). "Learning to predict where humans look." ICCV.
        Bylinskii et al. (2017). "What do different evaluation metrics
        tell us about saliency models?" IEEE TPAMI, arXiv:1604.03605.

    Parameters:
        prediction (numpy.ndarray): Predicted saliency map (the map being
            evaluated, e.g. from LIME or Integrated Gradients).
        reference (numpy.ndarray): Reference saliency map whose above-mean
            pixels define the fixation mask (e.g. a SHAP explanation or a
            prototype).

    Returns:
        float: AUC-Judd distance in [0, 1].  0 means perfect recovery of the
        reference's salient regions; 0.5 is chance level.
    """
    prediction = prediction.flatten().astype(np.float64)
    reference  = reference.flatten().astype(np.float64)

    fixations = reference >= reference.mean()
    return float(1.0 - _auc_judd_one_direction(prediction, fixations))


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
kl_divergence.__name__ = "KL Divergence"
information_gain.__name__ = "Information Gain"
nss.__name__ = "NSS"
linear_correlation_coefficient.__name__ = "Correlation Coefficient"
auc_judd.__name__ = "AUC"

# ------------- Set Theory metrics
jaccard_index.__name__ = "Jaccard Index"
jaccard_distance.__name__ = "Jaccard Distance"
czenakowski_distance.__name__ = "Czekanowski Distance"

# ------------- Binary metrics
sign_agreement_ratio.__name__ = "Sign Agreement Ratio"

# ------------- Structural metrics
ssim.__name__ = "SSIM"

# ------------- Information-theoretic metrics
kl_divergence.__name__ = "KL Divergence"
auc_judd.__name__ = "AUC-Judd"
