
# Diffuse Optical Imaging with CAFNet 

This repository contains the core implementation of the CAFNet model proposed in our manuscript "[Diffuse optical Imaging with CAFNet (channel attention fusion network)]" (JBO), an annotated simulated dataset, experimental phantom dataset as well as requirement file for readers.

<img width="1212" height="635" alt="CAFNet" src="https://github.com/user-attachments/assets/364043f1-45ea-43f1-9b20-0b2657f41463" />


---

## Table of Contents

- [Installation](#installation)  
- [Dataset](#dataset)  
- [Training](#training)  
- [Plotting Reconstructed Optical Properties](#plotting-reconstructed-optical-properties)  
- [Usage Notes](#usage-notes)  

---

## Installation

1. Make sure you have **Python 3.7.12** installed.  

2. Install the required dependencies:

```bash
pip install -r py-37.yaml
```
## Dataset
Ensure your dataset is in the correct folder structure expected by the scripts.

For test samples, the plotting utility uses a sample ID key (e.g., 'id_test_20_2d_exp') to select which data to visualize.

## Training
To train the CAFNet model, run:

```bash
python helper1516.py
```
This script will:

 1. Load the dataset

 2. Save the dataset in the directory 'trainingtata'

For generating plots after training, run:

```bash
python CAFNet_Model_Plot.py
```
## Plotting Reconstructed Optical Properties
To visualize the reconstructed optical-property images:

Open the plotting script or a Python session.

Change the key value to your desired test sample ID (Line #323):

```bash
python
for i in ['id_test_20_2d_exp']:  # Replace with your test sample ID
    plot_result(i, 'trainingtata', None)
```
Explanation of columns in the output plot:

Column 1: Ground truth optical properties

Column 2: Reconstructed results using Tikhonov Regularization

'trainingtata' refers to the directory containing the trained model outputs.

## Usage Notes
Make sure your dataset paths match the paths expected in the script




