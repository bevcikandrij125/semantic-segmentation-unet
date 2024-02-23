import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi


def mask_to_rle(mask):
    """
    Convert a binary mask to its run-length encoding representation.

    Parameters:
    - mask (numpy.ndarray): Binary mask array.

    Returns:
    - str: Encoded RLE string.
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) % 2 != 0:
        runs = np.append(runs, len(pixels))
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def download_and_unzip_kaggle_dataset(dataset_name, download_dir, kaggle_creds_path=None):
    """
    Download and unzip a dataset from Kaggle.

    Parameters:
    - dataset_name (str): Name of the dataset on Kaggle (e.g., 'username/dataset-name').
    - download_dir (str): Directory where the dataset should be downloaded and unzipped.
    - kaggle_creds_path (str, optional): Path to the Kaggle API credentials file.

    Returns:
    - bool: True if the download and unzip process was successful, False otherwise.
    """
    if kaggle_creds_path:
        api = KaggleApi(kaggle_creds_path=kaggle_creds_path)
    else:
        api = KaggleApi()

    try:
        api.authenticate()
    except Exception as e:
        print(f"Error authenticating with Kaggle API: {e}")
        return False

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    try:
        print("Downloading dataset...")
        api.dataset_download_files(dataset_name, path=download_dir, unzip=True)
        print("Download completed successfully.")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def check_directory(directory):
    """
    Check if a directory exists and is not empty.

    Parameters:
    - directory (str): Path to the directory.

    Returns:
    - bool: True if the directory exists and is not empty, False otherwise.
    """
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return False
    if not os.listdir(directory):
        print(f"The directory '{directory}' is empty.")
        return False
    return True


def load_and_preprocess_image(image_name, image_dir, target_size=None, normalize=True, transpose=False):
    """
    Load and preprocess an image from the specified directory.

    Parameters:
    - image_id (str): ID of the image file.
    - image_dir (str): Directory containing the image files.
    - target_size (tuple): Target size for the loaded image.
    - normalize (bool): Whether to normalize pixel values to [0, 1].
    - transpose (bool): Whether to transpose the image array.

    Returns:
    - numpy.ndarray: Preprocessed image array.
    """
    image_path = os.path.join(image_dir, image_name)
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    if normalize:
        image_array = image_array / 255.0
    if transpose:
        image_array = image_array.T
    return image_array


def generate_mask_from_encoded_pixels(encoded_pixels, image_shape):
    """
    Generate a mask from encoded pixels.

    Parameters:
    - encoded_pixels (str): Encoded pixel information in Run-Length Encoding (RLE) format.
    - image_shape (tuple): Shape of the original image (height, width).

    Returns:
    - numpy.ndarray: Binary mask representing the segmentation.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    encoded_pixels = str(encoded_pixels)
    if encoded_pixels == "nan":
        return mask
    pairs = encoded_pixels.split()
    for i in range(0, len(pairs), 2):
        start = int(pairs[i]) - 1
        length = int(pairs[i + 1])
        end = start + length
        start_row, start_col = divmod(start, image_shape[0])
        end_row, end_col = divmod(end, image_shape[0])
        mask[start_row:end_row + 1, start_col:end_col + 1] = 1
    return mask


def undersampling(train_df, nan_percentage):
    """
    Under samples the DataFrame based on the provided NaN percentage.

    Parameters:
    - train_df (pd.DataFrame): DataFrame containing training data.
    - nan_percentage (float): Percentage of NaN values.

    Returns:
    - pd.DataFrame: Under samples DataFrame.
    """
    non_nan_rows = train_df[train_df['EncodedPixels'].notna()]
    non_nan_count = len(non_nan_rows)
    nan_count = int(non_nan_count / (1 - nan_percentage)) - non_nan_count
    non_nan = non_nan_rows.sample(n=non_nan_count)
    undersampled_nan_rows = train_df[train_df['EncodedPixels'].isna()].sample(n=nan_count)
    undersampled_df = pd.concat([non_nan, undersampled_nan_rows], ignore_index=True)
    return undersampled_df


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - smooth (float): Smoothing factor.

    Returns:
    - float: Dice coefficient.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    """
    Computes the Dice coefficient loss.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - float: Dice coefficient loss.
    """
    return 1 - dice_coef(y_true, y_pred)


def recall4r1(y_true, y_pred):
    """
    Computes the recall for a given threshold.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - float: Recall for a given threshold.
    """
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())


def precision4r1(y_true, y_pred):
    """
    Computes the precision for a given threshold.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - float: Precision for a given threshold.
    """
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())


def plot_history(history):
    """
    Plots the training and validation loss, along with additional metrics if available.

    Parameters:
    - history: History object returned by model.fit().
    """
    # Get training history
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Get additional metrics if available
    metrics = ['accuracy', 'dice_score', 'precision4r1', 'recall4r1']
    metric_plots = []

    for metric in metrics:
        if metric in history.history:
            metric_plots.append((metric, history.history[metric]))

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot additional metrics
    for metric, values in metric_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(values, label=metric.capitalize())
        plt.title(metric.capitalize())
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.show()
