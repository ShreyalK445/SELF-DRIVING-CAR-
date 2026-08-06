#  Autonomous Driving

This repository contains an end-to-end deep learning pipeline for autonomous driving using behavioral cloning. The system maps raw RGB camera pixels directly to steering commands using a Convolutional Neural Network (CNN). 

Data collection is performed via the Udacity Self-Driving Car Simulator, and the model architecture is based on the NVIDIA End-to-End Learning paper.

## Architecture

The model is a standard feed-forward CNN designed for low-latency inference. 

- **Input:** 66x200x3 RGB images (cropped from the simulator's 160x320x3 output).
- **Normalization:** Lambda layer scaling pixel values to [-1, 1].
- **Convolutional Layers:** 5 layers (24, 36, 48, 64, 64 filters) with ELU activations and strided convolutions to reduce spatial dimensions.
- **Fully Connected Layers:** 3 dense layers (100, 50, 10).
- **Output:** Single continuous value representing the steering angle in radians.

## Data Pipeline

The simulator outputs a `driving_log.csv` alongside an `IMG/` directory. The training pipeline includes the following preprocessing steps:

1. **Cropping:** The top 60 pixels (sky/trees) and bottom 25 pixels (car hood) are removed. These regions contain no relevant features for lane tracking and only add noise.
2. **Resizing:** Images are resized to 200x66 to match the NVIDIA architecture constraints and reduce computational overhead.
3. **Augmentation:** To prevent the model from biasing toward left or right turns (depending on the track layout), 50% of the training images are randomly flipped horizontally, and their steering angles are inverted.

## Setup and Installation

Requires Python 3.8+.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
