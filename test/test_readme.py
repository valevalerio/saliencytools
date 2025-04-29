from saliencytools import ssim, psnr, emd

import numpy as np
import matplotlib.pyplot as plt


def test_readme():
    """
    This function is used to test the README file
    by running the example code in the README file
    """
    # create a random saliency map
    saliency_map = np.random.rand(28*28).reshape(28, 28)
    # create a random ground truth map
    ground_truth_map = np.random.rand(28*28).reshape(28, 28)

    # use all the metrics to compare the saliency map with the ground truth map
    for metric in [ssim, psnr, emd]:
        print(f"{metric.__name__}: {metric(saliency_map, ground_truth_map)}")
