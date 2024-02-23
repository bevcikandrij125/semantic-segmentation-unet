import os

PROJECT_DIR = ""

# KAGGLE_CREDS_PATH or DATA_PATH is required
KAGGLE_CREDS_PATH = os.path.join(PROJECT_DIR, "kaggle.json")
DATASET_NAME = 'airbus-ship-detection'

LOGS_DIR_PATH = os.path.join(PROJECT_DIR, 'logs')
WEIGHTS_DIR_PATH = os.path.join(PROJECT_DIR, "weights")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")

DATA_PATH = os.path.join(PROJECT_DIR, 'data')

TRAIN_PATH = os.path.join(DATA_PATH, 'train_v2')
TEST_PATH = os.path.join(DATA_PATH, 'test_v2')

TRAIN_LABELS_PATH = os.path.join(DATA_PATH, "train_ship_segmentations_v2.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_PATH, "sample_submission_v2.csv")

TARGET_IMAGE_SHAPE = (256, 256, 3)

"""
Project Directory (PROJECT_DIR): Set the PROJECT_DIR variable to the root directory of your project.

Kaggle Credentials Path (KAGGLE_CREDS_PATH): Specify the path to your Kaggle API credentials JSON file.
This is required if you intend to download datasets from Kaggle.

Data Paths (DATA_PATH, TRAIN_PATH, TEST_PATH, TRAIN_LABELS_PATH, SAMPLE_SUBMISSION_PATH):
Specify paths to different parts of your dataset, such as training images, test images, training labels, and sample submission files.
If you have pre-existing data, set DATA_PATH to the directory containing your dataset.
If DATA_PATH is not provided or the directory does not exist, the script will attempt to download the  DATASET_NAME dataset  using the Kaggle API credentials specified in KAGGLE_CREDS_PATH and store it in DATA_PATH.


Dataset Name (DATASET_NAME): Specify the name of the dataset you're working with.

Directories and Paths: Define paths to various directories and files within your project, such as logs, weights, models, and data directories.

Target Image Shape (TARGET_IMAGE_SHAPE): Set the target shape for your input images, typically in the format (height, width, channels).
"""
