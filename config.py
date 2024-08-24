"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# Standard Library imports
from pathlib import Path

# External imports
import tensorflow as tf


# Data paths
DATASET = Path(r"<<FILL ME>>")
TRAIN_DATASET = DATASET / "train"
VALID_DATASET = DATASET / "val"

QUERY_DATASET = Path(r"<<FILL ME>>")

# Model input image size
IMAGE_SIZE = (<<FILL ME>>, <<FILL ME>>)

AUTO = tf.data.AUTOTUNE

# Training parameters
LEARNING_RATE = 0.0001
INITIAL_EPOCH = 0
INITIAL_VALUE_THRESH = None  # Set to None if first training
EPOCHS = 100
BATCH_SIZE = 4

# Output paths
OUTPUT_PATH = Path("siamese_output")
MODEL_PATH = OUTPUT_PATH / "siamese_model.keras"
MODEL_CKPT_PATH = OUTPUT_PATH / "epoch_{epoch:02d}-loss_{val_loss:.4f}.keras"
INDEX_PATH = OUTPUT_PATH / "index.faiss"
LOGS_PATH = OUTPUT_PATH / "logs"