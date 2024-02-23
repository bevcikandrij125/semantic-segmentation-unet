# Semantic Segmentation with UNet

This repository contains code for training a UNet model for semantic segmentation tasks using TensorFlow. The model is trained on the Airbus Ship Detection dataset and is capable of segmenting ships in satellite images.

## Prerequisites

- Python 3
- TensorFlow
- Kaggle API (for downloading the dataset)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/bevcikandrij125/semantic-segmentation-unet.git
   cd semantic-segmentation-unet
2.  Install the required dependencies:
    
    bashCopy code

    `pip install -r requirements.txt`
    
Configuration
-------------

The `config.py` file contains configuration parameters such as file paths, dataset name, and Kaggle API credentials. 
If DATA_PATH is not provided or the directory does not exist, the script will attempt to download the  DATASET_NAME dataset  using the Kaggle API credentials specified in KAGGLE_CREDS_PATH and store it in DATA_PATH.
Ensure that these parameters are correctly set before running the scripts.

Usage
-----

### 1\. Data Preparation

Run `preprocessing.py` to prepare the dataset. This script will download the Airbus Ship Detection dataset if it's not already present and split it into training, testing, and validation sets. It will save the split datasets into CSV files (`train.csv`, `test.csv`, `val.csv`).

bashCopy code

`python preprocessing.py`

### 2\. Model Training

Run `train.py` to train the UNet model using the prepared dataset. This script will load the datasets, define and compile the UNet model, train the model, and save the trained model weights.

bashCopy code

`python train.py`

### 3\. Inference

Run `inference.py` to perform inference on the test set images. This script will load the trained model, perform inference, convert the predicted masks to Run-Length Encoding (RLE) format, and create a submission CSV file (`submission.csv`) with the predicted masks.

bashCopy code

`python inference.py`

