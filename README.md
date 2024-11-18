# BrainGridNet

**BrainGridNet** is a dual-branch depthwise convolutional neural network developed for decoding multi-class motor imagery from EEG signals. This repository provides the implementation of our experimental models, **BrainGridNet_PSD32** and **BrainGridNet_Raw**, as well as comparative models used for ablation experiments.

---

## Features
- Supports both **PSD** and **raw EEG data** as inputs.
- Achieves high performance in decoding multi-class motor imagery.
- Provides ablation models to evaluate the contribution of different components.

---

## Getting Started

### Prerequisites
Ensure the following dependencies are installed before using this repository:
- **Python**: 3.9 or higher  
- **PyTorch**: torch 2.2.2 + cu121  
- **NumPy**: 1.23.2  

---

### Usage

#### Input and Output Shapes
For **PSD** input:
- **Input Shape**: `(batch_size, 9, 9, 32)`  
  _(Note: Apply `10 * np.log10()` to the PSD data to ensure effective feature learning. For shape details, refer to the paper.)_
- **Output Shape**: `(batch_size, num_classes)`

For **Raw** input:
- **Input Shape**: `(batch_size, 9, 9, 655)`  
- **Output Shape**: `(batch_size, num_classes)`

#### Model Descriptions
- **`BrainGridNet_PSD32`**: Processes PSD data of EEG signals.
- **`BrainGridNet_Raw`**: Processes raw EEG data.

---

### Example Usage
```python
# Example code to load and run BrainGridNet_PSD32 or BrainGridNet_Raw
from BrainGridNet import BrainGridNet_PSD32, BrainGridNet_Raw

# Load your data (example shape)
input_data_psd = ...  # Shape: (batch_size, 9, 9, 32)
input_data_raw = ...  # Shape: (batch_size, 9, 9, 655)

# Instantiate the model for PSD input
model_psd = BrainGridNet_PSD32()
output_psd = model_psd(input_data_psd)

# Instantiate the model for Raw input
model_raw = BrainGridNet_Raw()
output_raw = model_raw(input_data_raw)