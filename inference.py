import config
import numpy as np
import pandas as pd
import tensorflow as tf
from helping_functions import load_and_preprocess_image, mask_to_rle


test_df = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
model = tf.keras.models.load_model(config.MODEL_DIR)
first_id = test_df.iloc[0]["ImageId"]
original_image_shape = load_and_preprocess_image(first_id, config.TEST_PATH).shape


def inference(image_name):
    image = load_and_preprocess_image(image_name, config.TEST_PATH)
    # Expand dimensions to make it a batch of 1
    image = np.expand_dims(image, axis=0)
    predicted_mask = model.predict(image)
    return predicted_mask


predictions = []
for image_id in test_df['ImageId']:
    mask = inference(image_id)
    resized_mask = tf.image.resize(mask, original_image_shape)
    rle_encoded = mask_to_rle(resized_mask)
    predictions.append(rle_encoded)


submission_df = pd.DataFrame({
    'ImageId': test_df['ImageId'],
    'EncodedPixels': predictions
})
submission_df.to_csv('submission.csv', index=False)
