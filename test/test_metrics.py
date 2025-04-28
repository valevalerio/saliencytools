""" This Module is used to check the coherency of the metrics implemented in the project."""

from saliencytools import (
    normalize_mask_0_1,
    clip_mask,
    euclidean_distance,
    cosine_distance,
    emd,
    mean_absolute_error,
    sign_agreement_ratio,
    sign_distance,
    intersection_over_union,
    correlation_distance,
    mean_squared_error,
    ssim,
    psnr,
    czenakowski_distance,
    jaccard_distance)
import numpy as np
def test_metrics():
    
    test_map_1 = np.random.rand(28*28).reshape(28, 28)
    test_map_2 = np.random.rand(28*28).reshape(28, 28)
    
    # Clip the maps
    test_map_1 = clip_mask(test_map_1)
    test_map_2 = clip_mask(test_map_2)
    # Normalize the maps
    test_map_1 = normalize_mask_0_1(test_map_1)
    test_map_2 = normalize_mask_0_1(test_map_2)

    # Check if the maps are not empty
    assert test_map_1.size > 0, "test_map_1 is empty"
    assert test_map_2.size > 0, "test_map_2 is empty"
    
    for metric in [
        euclidean_distance,
        cosine_distance,
        emd,
        mean_absolute_error,
        sign_agreement_ratio,
        sign_distance,
        intersection_over_union,
        correlation_distance,
        mean_squared_error,
        ssim,
        psnr,
        czenakowski_distance,
        jaccard_distance
    ]:
        # Check if the metric is symmetric
        assert metric(test_map_1, test_map_2) == metric(test_map_2, test_map_1), f"{metric.__name__} is not symmetric"
        # Check if the metric is non-negative
        assert metric(test_map_1, test_map_2) >= 0, f"{metric.__name__} is negative"
        # Check if the metric is zero when both maps are identical
        if metric.__name__ not in ["ssim", "psnr"]:
            # Skip ssim and psnr for this test
            # as they are not zero when the maps are identical
            continue
        # Check if the metric is zero when both maps are identical
        assert np.allclose(metric(test_map_1, test_map_1), 0), f"{metric.__name__} is not zero when maps are identical"
