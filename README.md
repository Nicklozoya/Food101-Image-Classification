# Food101 Classification with Transfer Learning

This project implements a deep learning model to classify images from the [Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) using transfer learning and fine-tuning with TensorFlow. The model achieves an accuracy of **78.26%** on the test set using EfficientNetB0.

## Overview

- **Dataset**: Food101 (101 classes, 75,750 training, 25,250 validation images)
- **Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Accuracy**: 78.26%
- **Techniques**: Feature extraction, fine-tuning, mixed precision training


## Prerequisites

- Python 3.8+
- TensorFlow (nightly recommended)
- NVIDIA GPU (optional)
- Install: `pip install -U -q tf-nightly`

## Usage

1. Clone this repository.
2. Run the code in a Jupyter Notebook or Python script (see project code for details).
3. Ensure a GPU is available for optimal performance.

## Results

- **Test Accuracy**: 78.26%

## Future Improvements

- Try larger models (e.g., EfficientNetB4)

## Acknowledgments

- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
