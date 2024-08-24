"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# Standard Library imports
import random
from pathlib import Path

# External imports
import tensorflow as tf
from tensorflow.keras import Sequential
import tensorflow.keras.layers as layers


class AugmentMapFunction:
    def __init__(self):
        self.trainAug = Sequential(
            [
                layers.RandomBrightness(0.2, value_range=(0, 1)),
                layers.RandomContrast(0.2),
                layers.RandomFlip("vertical"),
                layers.RandomZoom(
                    height_factor=(-0.05, -0.15), width_factor=(-0.05, -0.15), fill_mode="constant"
                ),
                layers.RandomRotation(0.01, fill_mode="constant"),
                layers.RandomTranslation(height_factor=0.15, width_factor=0.15, fill_mode="constant"),
            ]
        )

    def __call__(self, anchor, positive, negative):
        positive = self.trainAug(positive)
        return (anchor, positive, negative)


class MapFunction:
    def __init__(self, image_size):
        self.image_size = image_size

    def decode_and_resize(self, imagePath):
        image = tf.io.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Does rescaling by 1/255!
        image = tf.image.resize(image, self.image_size)
        return image

    def __call__(self, anchor, negative):
        anchor = self.decode_and_resize(anchor)
        positive = tf.identity(anchor)
        negative = self.decode_and_resize(negative)
        return (anchor, positive, negative)


class PairsGenerator:
    def __init__(self, datasetPath: Path):
        self.filepaths = [str(f) for f in datasetPath.rglob("*.jpg")]

    def get_next_element(self):
        while True:
            anchor = random.choice(self.filepaths)
            temp = self.filepaths.copy()
            temp.remove(anchor)
            negative = random.choice(temp)
            yield (anchor, negative)
