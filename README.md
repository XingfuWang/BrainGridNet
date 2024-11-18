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
  _(Note: The TimesNet method plays a critical role in effectively learning features from raw EEG signals. It helps capture temporal dependencies and enhances model performance. See the figure below for the architecture of TimesNet.)_
- **Output Shape**: `(batch_size, num_classes)`

#### Model Descriptions
- **`BrainGridNet_PSD32`**: Processes PSD data of EEG signals.
- **`BrainGridNet_Raw`**: Processes raw EEG data.

### TimesNet Architecture
Below is the architecture of the TimesNet module used in `BrainGridNet_Raw` to process raw EEG signals:

![TimesNet Architecture](https://github.com/XingfuWang/BrainGridNet/blob/main/Timesnet_method.png)
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
```

## Citation

If you find this repository helpful, please consider citing our work:

```makefile
@article{10.1016/j.neunet.2023.11.037,
author = {Wang, Xingfu and Wang, Yu and Qi, Wenxia and Kong, Delin and Wang, Wei},
title = {BrainGridNet: A two-branch depthwise CNN for decoding EEG-based multi-class motor imagery},
year = {2024},
issue_date = {Feb 2024},
publisher = {Elsevier Science Ltd.},
address = {GBR},
volume = {170},
number = {C},
issn = {0893-6080},
url = {https://doi.org/10.1016/j.neunet.2023.11.037},
doi = {10.1016/j.neunet.2023.11.037},
journal = {Neural Netw.},
month = apr,
pages = {312–324},
numpages = {13},
keywords = {Convolutional Neural Network (CNN), Power Spectral Density (PSD), Electroencephalogram (EEG), Multi-class motor imagery, Computational costs}
}
```
