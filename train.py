"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# External imports
from tensorflow import keras
import tensorflow as tf

# Local imports
from siamese.dataset import PairsGenerator
from siamese.dataset import MapFunction, AugmentMapFunction
from siamese.model import get_embedding_module
from siamese.model import get_siamese_network
from siamese.model import SiameseModel
import siamese.config as config


def prepare(ds, shuffle=False, augment=False):
    """
    From:
    https://www.tensorflow.org/tutorials/images/data_augmentation
    """

    # Resize and rescale all datasets
    ds = ds.map(common_map_fun, num_parallel_calls=config.AUTO)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets
    ds = ds.batch(config.BATCH_SIZE)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(aug_map_fun, num_parallel_calls=config.AUTO)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=config.AUTO)


train_generator = PairsGenerator(datasetPath=config.TRAIN_DATASET)
valid_generator = PairsGenerator(datasetPath=config.VALID_DATASET)

train_dataset = tf.data.Dataset.from_generator(
    generator=train_generator.get_next_element,
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
    ),
)

valid_dataset = tf.data.Dataset.from_generator(
    generator=valid_generator.get_next_element,
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
    ),
)


common_map_fun = MapFunction(image_size=config.IMAGE_SIZE)
aug_map_fun = AugmentMapFunction()

train_ds = prepare(train_dataset, shuffle=True, augment=True)
valid_ds = prepare(valid_dataset)

embedding_module = get_embedding_module(image_size=config.IMAGE_SIZE)
siamese_net = get_siamese_network(
    image_size=config.IMAGE_SIZE,
    embedding_model=embedding_module,
)
siamese_model = SiameseModel(
    siamese_net=siamese_net,
    margin=0.5,
    lossTracker=keras.metrics.Mean(name="loss"),
)
siamese_model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE))

print("Training the siamese model...")
siamese_model.fit(
    train_ds,
    steps_per_epoch=config.STEPS_PER_EPOCH,
    validation_data=valid_ds,
    validation_steps=config.VALIDATION_STEPS,
    epochs=config.EPOCHS,
)


print(f"Saving the siamese network to {config.MODEL_PATH}...")
config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
keras.models.save_model(
    model=siamese_model.siamese_net,
    filepath=config.MODEL_PATH,
    include_optimizer=False,
)
