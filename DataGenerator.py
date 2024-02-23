import numpy as np
import tensorflow as tf
from helping_functions import load_and_preprocess_image, generate_mask_from_encoded_pixels


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, im_dir, data_df, batch_size=32, target_image_shape=None):
        self.data_df = data_df
        self.im_dir = im_dir
        self.batch_size = batch_size

        first_id = self.data_df.iloc[0]["ImageId"]
        original_image_shape = load_and_preprocess_image(first_id, self.im_dir).shape
        self.original_image_size = original_image_shape[:2]
        self.n_colour_channels = original_image_shape[2]
        if target_image_shape is None:
            self.target_size = self.original_image_size
        else:
            self.target_size = target_image_shape[:2]

    def __len__(self):
        return int(len(self.data_df) / self.batch_size)

    def on_epoch_end(self):
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.data_df))

        df = self.data_df.iloc[start:end]
        batch_images_list = []
        batch_masks_list = []
        for index, row in df.iterrows():
            image = np.zeros((self.batch_size,) + self.target_size + (self.n_colour_channels,))
            resized_mask = np.zeros((self.batch_size,) + self.target_size + (1,))
            try:
                image = load_and_preprocess_image(row["ImageId"], self.im_dir, self.target_size)
                mask = generate_mask_from_encoded_pixels(row["EncodedPixels"], self.original_image_size)
                mask_with_extra_dim = tf.expand_dims(mask, axis=-1)
                resized_mask = tf.image.resize(mask_with_extra_dim, self.target_size)
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                image = np.zeros((self.batch_size,) + self.target_size + (self.n_colour_channels,))
                resized_mask = np.zeros((self.batch_size,) + self.target_size + (1,))
            finally:
                batch_images_list.append(image)
                batch_masks_list.append(resized_mask)

        return np.array(batch_images_list), np.array(batch_masks_list)
