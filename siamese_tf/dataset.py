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

import cv2

# Local imports
import config as config
from augmentations import al_augmentations
from utils import get_image_paths


def random_vertical_flip(image, p=0.5):
    """From: https://stackoverflow.com/a/69689352"""
    if random.random() < p:
        return tf.image.random_flip_up_down(image)
    return image


def apply_albumentations(image):
    data = {"image": image}
    aug_data = al_augmentations(**data)
    aug_img = aug_data["image"]
    return aug_img


tf_augmentations = Sequential(
    [
        # layers.Identity(),
        layers.RandomBrightness(factor=(-0.2, 0.2), value_range=(0, 1)),
        layers.RandomContrast(factor=(0.0, 0.2)),
        layers.Lambda(random_vertical_flip, arguments={"p": 0.01}),
    ]
)


class AugmentMapFunction:
    def __init__(self):
        pass

    def apply_albumentations(self, image):
        return tf.numpy_function(
            func=apply_albumentations, inp=[image], Tout=tf.float32
        )

    def __call__(self, anchor, positive, negative):
        positive = tf_augmentations(positive)
        positive = tf.numpy_function(
            func=apply_albumentations, inp=[positive], Tout=tf.float32
        )
        positive.set_shape(anchor.get_shape())
        return (anchor, positive, negative)


class CommonMapFunction:
    def __init__(self, image_size):
        self.image_size = image_size

    def decode_and_resize(self, imagePath):
        image = tf.io.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(
            image, dtype=tf.float32
        )  # Does rescaling by 1/255!
        image = tf.image.resize(image, self.image_size)
        return image

    def __call__(self, anchor, negative):
        anchor = self.decode_and_resize(anchor)
        positive = tf.identity(anchor)
        negative = self.decode_and_resize(negative)
        return (anchor, positive, negative)


class PairsGenerator:
    def __init__(self, dataset: Path):
        self.filepaths = get_image_paths(dataset, return_str=True)
        self.total_files = len(self.filepaths)
        self.index = 0

    def get_next_element(self):
        """ """

        while True:

            if self.index >= self.total_files:
                self.index = 0

            anchor = self.filepaths[self.index]
            self.index += 1

            # Since there is only one element per class,
            # a negative element will be any other random element
            temp = self.filepaths.copy()
            temp.remove(anchor)
            negative = random.choice(temp)

            yield (anchor, negative)


def create_dataset(generator):
    return tf.data.Dataset.from_generator(
        generator=generator.get_next_element,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )


def prepare_dataset(ds, common_map, aug_map, shuffle=False, augment=False):
    """ """

    ds = ds.map(common_map, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    if augment:
        ds = ds.map(aug_map, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(config.BATCH_SIZE)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)
