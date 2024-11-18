# BrainGridNet

BrainGridNet is a dual-branch depthwise convolutional neural network designed for decoding multi-class motor imagery from EEG signals. This repository provides the implementation of our experimental models, **BrainGridNet_PSD32** and **BrainGridNet_Raw**, along with comparative models used for ablation experiments.

## Features
- Designed for PSD and raw EEG data.
- Demonstrates high performance in decoding multi-class motor imagery.
- Includes models tailored for ablation studies to evaluate the contribution of various components.

## Getting Started

### Prerequisites
Before using this repository, ensure you have the following dependencies installed:
- Python (>= 3.7)
- TensorFlow/PyTorch (mention the framework used)
- NumPy
- (Any additional libraries)

### Usage

#### Input and Output Shape
- **Input Shape:** `(batch_size, 9, 9, 32)` (For PSD data, `10 * np.log10()` transformation is required; refer to our paper for shape details.)
- **Output Shape:** `(batch_size, num_classes)`

#### Model Descriptions
- `BrainGridNet_PSD32` processes PSD data of EEG signals.
- `BrainGridNet_Raw` is designed for raw EEG data processing.

### Example
```python
# Example code to load and run BrainGridNet_PSD32
from model import BrainGridNet_PSD32

# Load your data (example shape)
input_data = ...  # Shape: (batch_size, 9, 9, 32)

# Instantiate the model
model = BrainGridNet_PSD32()

# Forward pass
output = model(input_data)
