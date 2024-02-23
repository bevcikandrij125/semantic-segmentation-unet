import os
import config
import pandas as pd
import tensorflow as tf
from Unet import UNetModel
from DataGenerator import DataGenerator
from helping_functions import dice_coef_loss, recall4r1, precision4r1


train_df = pd.read_csv(os.path.join(config.PROJECT_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(config.PROJECT_DIR, 'test.csv'))
val_df = pd.read_csv(os.path.join(config.PROJECT_DIR, 'val.csv'))

batch_size = 32
steps_per_epoch = len(train_df) // batch_size
image_shape = config.TARGET_IMAGE_SHAPE  # (256, 256, 3)

train_generator = DataGenerator(config.TRAIN_PATH, train_df, batch_size, image_shape)
val_generator = DataGenerator(config.TRAIN_PATH, val_df, batch_size, image_shape)

model = UNetModel(image_shape)
model.compile(optimizer='adam',
              loss=dice_coef_loss,
              metrics=[recall4r1, precision4r1])

log_dir = os.path.join(config.LOGS_DIR_PATH, "UNet1")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

checkpoint_filename = "Ep{epoch:03d}.h5"
checkpoint_filepath = os.path.join(config.WEIGHTS_DIR_PATH, checkpoint_filename)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=checkpoint_filepath,
  save_weights_only=True,
  save_freq="epoch"
)

history = model.fit(train_generator,
                    validation_data=val_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=15,
                    batch_size=batch_size,
                    callbacks=[tensorboard_callback, early_stopping_callback, model_checkpoint_callback],
                    workers=tf.data.AUTOTUNE,
                    use_multiprocessing=True)
model.save(config.MODEL_DIR)
