Usage
=====

Here is an example of how to use the metrics provided by **saliencytools**:

.. code-block:: python

   from saliencytools import ssim, psnr, emd
   import numpy as np

   # Create random saliency and ground truth maps
   saliency_map = np.random.rand(28, 28)
   ground_truth_map = np.random.rand(28, 28)

   # Compare using metrics
   for metric in [ssim, psnr, emd]:
       print(f"{metric.__name__}: {metric(saliency_map, ground_truth_map)}")

.. todo:: Add more detailed examples and visualizations.
