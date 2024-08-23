"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# Standard Library imports
from pathlib import Path

# External imports
import tensorflow as tf


# path to training and testing data
TRAIN_DATASET = Path(<<FILL ME>>)
VALID_DATASET = Path(<<FILL ME>>)

# model input image size
IMAGE_SIZE = (680, 488)

# batch size and the buffer size
BATCH_SIZE = 4
BUFFER_SIZE = BATCH_SIZE * 2

# define autotune
AUTO = tf.data.AUTOTUNE

# define the training parameters
LEARNING_RATE = 0.0001
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 10
EPOCHS = 10

# define the path to save the model
OUTPUT_PATH = Path("siamese_output")
MODEL_PATH = OUTPUT_PATH / "siamese_network.keras"
OUTPUT_IMAGE_PATH = OUTPUT_PATH / "output_image.png"
