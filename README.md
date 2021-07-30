# Segmentation Architectures

This repository contains multiple architectures for segmentation. The architectures are:
- UNet (from https://arxiv.org/abs/1505.04597) with a lot of additions, which include
  - Residual connections
  - Attention
  - Squeeze and Excitation blocks
  - CBAM blocks
- Deeplabv3+ (from https://arxiv.org/abs/1802.02611v2)
- DeepTiramisu (from https://arxiv.org/abs/1611.09326)

## Getting Started

The models can be build by calling the build_model function in the submodule corresponding to the model.

### Prerequisites

No prerequisites are required besides the modules listed in the requirements.txt file.

### Installing

It is best to use virtualenv to create a virtual environment

python -m virtualenv venv

And then install the requirements.

Pre-commit can be installed with

pip install pre-commit

The cooks will be installed by

pre-commit install

You can run the hooks for all files using (usually, they are run only for files being committed)

pre-commit run --all-files

## Running the tests

- The test can be run using pytest
- They can also be run by hand using python -m SegmentationArchitectures.test_architectures

### Running the training

For training, just compile the model and train it with model.fit or a custom training function (easiest way is to subclass the tf model)