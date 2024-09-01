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

# Local imports
import siamese.config as config

WHITE = (1, 1, 1)


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

al_augmentations = A.Compose(
    [
        A.Blur(blur_limit=5),
        A.CoarseDropout(p=0.1),
        # Zoom
        A.ShiftScaleRotate(
            shift_limit=0,
            rotate_limit=0,
            scale_limit=(-0.2, 0),  # Zoom out only
            border_mode=cv2.BORDER_CONSTANT,
            value=WHITE,
            p=1.0,
        ),
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
        A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=WHITE),
    ]
)


class AugmentMapFunction:

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


def get_image_paths(folder: Path, return_str=False) -> list[Path | str]:
    """Get all the image paths from a folder"""
    paths = []
    for ext in config.EXTENSIONS:
        folder_paths = list(folder.rglob(ext))
        if return_str:
            folder_paths = [str(f) for f in folder_paths]
        paths.extend(folder_paths)
    return paths


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

    ds = ds.map(common_map, num_parallel_calls=config.AUTO)

    if shuffle:
        ds = ds.shuffle(1000)

    if augment:
        ds = ds.map(aug_map, num_parallel_calls=config.AUTO)

    ds = ds.batch(config.BATCH_SIZE)

    return ds.prefetch(buffer_size=config.AUTO)
