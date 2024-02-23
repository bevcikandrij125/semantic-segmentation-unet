import os
import tensorflow as tf


class UNetModel(tf.keras.Model):
    def __init__(self, input_shape, weights_path=None, units=32, kernel_size=3, blocks=4, conv_in_block_up=2,
                 conv_in_block_down=2, num_classes=2):
        super(UNetModel, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.conv_in_block_up = conv_in_block_up
        self.conv_in_block_down = conv_in_block_down
        self.num_classes = num_classes

        self.inputs = tf.keras.layers.Input(input_shape)
        self.conv_layers = []

        # Encoder
        x = self.inputs
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        for _ in range(blocks):
            for _ in range(conv_in_block_down):
                x = tf.keras.layers.Conv2D(units, kernel_size, activation='relu', padding='same')(x)
            self.conv_layers.append(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            units *= 2

        # Bridge
        x = tf.keras.layers.Conv2D(units, kernel_size, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(units, kernel_size, activation='relu', padding='same')(x)

        # Decoder
        for _ in range(blocks):
            units //= 2
            x = tf.keras.layers.Conv2DTranspose(units, (2, 2), strides=(2, 2), padding='same')(x)
            x = tf.keras.layers.concatenate([x, self.conv_layers.pop()], axis=3)
            for _ in range(conv_in_block_up):
                x = tf.keras.layers.Conv2D(units, kernel_size, activation='relu', padding='same')(x)

        if num_classes == 2:
            self.outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
        else:
            self.outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(x)

        if weights_path is not None and os.path.exists(weights_path):
            self.load_weights(weights_path)
