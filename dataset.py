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
import albumentations as A
import cv2

WHITE = (1, 1, 1)

transforms = A.Compose(
    [
        A.Perspective(
            fit_output=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=WHITE, p=0.3
        ),
        
        # Shift only
        A.ShiftScaleRotate(
            shift_limit=0.05,
            rotate_limit=0,
            scale_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            value=WHITE,
            p=0.5,
        ),

        A.SafeRotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=WHITE, p=0.1),
    ]
)


def album_augmentations(image):
    data = {"image": image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    return aug_img


def random_vertical_flip(image, p=0.5):
    """From: https://stackoverflow.com/a/69689352"""
    if random.random() < p:
        return tf.image.random_flip_up_down(image)
    return image


class AugmentMapFunction:
    def __init__(self):
        self.tf_augmentations = Sequential(
            [
                # layers.Identity(),
                layers.RandomBrightness(factor=(-0.2, 0.2), value_range=(0, 1)),
                layers.RandomContrast(factor=(0.0, 0.2)),
                layers.Lambda(random_vertical_flip, arguments={"p": 0.01}),
                layers.RandomZoom(
                    height_factor=(0.0, 0.2),
                    width_factor=(0.0, 0.2),
                    fill_mode="constant",
                    fill_value=1.0,
                ),
            ]
        )

    def __call__(self, anchor, positive, negative):
        positive = self.tf_augmentations(positive)
        positive = tf.numpy_function(
            func=album_augmentations, inp=[positive], Tout=tf.float32
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
    def __init__(self, datasetPath: Path):
        self.filepaths = [str(f) for f in datasetPath.rglob("*.jpg")]

    def get_next_element(self):
        while True:
            anchor = random.choice(self.filepaths)
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
