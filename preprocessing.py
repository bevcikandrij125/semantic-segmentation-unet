import os
import config
import pandas as pd
from sklearn.model_selection import train_test_split
from helping_functions import check_directory, download_and_unzip_kaggle_dataset

if not check_directory(config.DATA_PATH):
    download_and_unzip_kaggle_dataset(config.DATASET_NAME, config.DATA_PATH, config.KAGGLE_CREDS_PATH)


def is_image_exists(image_id):
    return os.path.exists(os.path.join(config.TRAIN_PATH, image_id))


def merge(image_id, df):
    image_df = df[df["ImageId"] == image_id]
    # Convert "EncodedPixels" values to strings and filter out any NaN values
    encoded_pixels = image_df["EncodedPixels"].astype(str).dropna()
    combined_mask = " ".join(encoded_pixels)
    return combined_mask


seg_df = pd.read_csv(config.TRAIN_LABELS_PATH)
image_ids = seg_df[seg_df.apply(lambda x: is_image_exists(x["ImageId"]), axis=1)]["ImageId"].unique()
image_ids_series = pd.Series(image_ids)
labels = image_ids_series.apply(lambda x: merge(x, seg_df))

train_ids, test_ids, train_labels, test_labels = train_test_split(image_ids, labels, test_size=0.15, random_state=42)
test_ids, val_ids, test_labels, val_labels = train_test_split(test_ids, test_labels, test_size=0.5, random_state=42)

train_df = pd.DataFrame({'ImageId': train_ids, 'EncodedPixels': train_labels})
test_df = pd.DataFrame({'ImageId': test_ids, 'EncodedPixels': test_labels})
val_df = pd.DataFrame({'ImageId': val_ids, 'EncodedPixels': val_labels})

train_df.to_csv(os.path.join(config.PROJECT_DIR, 'train.csv'), index=False)
test_df.to_csv(os.path.join(config.PROJECT_DIR, 'test.csv'), index=False)
val_df.to_csv(os.path.join(config.PROJECT_DIR, 'val.csv'), index=False)
