"""
main.py

Entry point for running the full pipeline of cine MRI segmentation experiments.
This script provides a unified interface for:

- Dataset preparation (k-space, fully-sampled images, undersampled images).
- Training of different segmentation models:
  * K-space only
  * Image only
  * Hybrid K-space + Image
  (for both fully-sampled and undersampled cases).
- Evaluation of trained models on validation data.
- Visualization of segmentation results.

Users can comment/uncomment the corresponding function calls in `main()` to
select which stage(s) of the pipeline to run. This allows flexible switching
between data preparation, training, evaluation, and visualization.

Note
----
On first execution, the three dataset preparation functions should be run first
(`prepare_k_space()`, `prepare_image_full()`, and `prepare_image_under()`).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from process.evaluation import K_full_evaluate, image_full_evaluate, K_under_evaluate, K_image_full_evaluate, \
    image_under_evaluate, K_image_under_evaluate
from process.image_model_training import image_under_train, image_full_train
from process.k_image_model_training import K_image_under_train, K_image_full_train
from process.k_model_training import K_under_train, K_full_train
from process.prepare_data import prepare_k_space, prepare_image_full, prepare_image_under
from process.visualisation import image_full_vis, image_under_vis, K_full_vis, K_under_vis
from settings import PATH_K, PATH_FULL_OUT, PATH_UNDER_OUT
from utils.main_utils import set_seed



def main():
    ## check or update settings before running (settings.py)

    ## prepare the dataset
    # prepare_k_space(PATH_K)
    # prepare_image_full(PATH_K, PATH_FULL_OUT)
    # prepare_image_under(PATH_K, PATH_UNDER_OUT)

    ## train the models (fully sampled + undersampled)
    # when training using the fully sampled data, this number should be 3
    # K_full_train()
    # image_full_train()
    # K_image_full_train()
    # K_under_train()
    # image_under_train()
    # K_image_under_train()

    ## evaluate the model
    # K_full_evaluate()
    # image_full_evaluate()
    K_image_full_evaluate()
    # K_under_evaluate()
    # image_under_evaluate()
    # K_image_under_evaluate()

    ## visualise results (example)
    # K_full_vis(index=60)
    # image_under_vis(index=60)


if __name__ == "__main__":
    main()


