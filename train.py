"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# External imports
import tensorflow as tf
import matplotlib.pyplot as plt

# Local imports
from siamese.dataset import PairsGenerator
from siamese.dataset import MapFunction, AugmentMapFunction
from siamese.model import get_embedding_module
from siamese.model import get_siamese_network
from siamese.model import SiameseModel
import siamese.config as config
from siamese.create_index import create_index


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


def visualize_triplets(dataset, n_batches=1):
    """ """
    
    for i, (anchors, positives, negatives) in enumerate(dataset):
        
        if i == n_batches:
            break
        
        fig = plt.figure(figsize=(12, 8))
        ax1, ax2, ax3 = fig.subplots(nrows=3, ncols=config.BATCH_SIZE)

        for i in range(0, config.BATCH_SIZE):
            anchor_im = anchors[i].numpy()
            positive_im = positives[i].numpy()
            negative_im = negatives[i].numpy()
            
            ax1[i].imshow(anchor_im)
            ax2[i].imshow(positive_im)
            ax3[i].imshow(negative_im)

            plt.axis("off")

        plt.tight_layout()
        plt.show()

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
valid_ds = prepare(valid_dataset, augment=True)

visualize_triplets(train_ds, n_batches=1)

if config.MODEL_PATH.exists():
    print(f"Loading model {config.MODEL_PATH}")
    siamese_model = tf.keras.models.load_model(filepath=config.MODEL_PATH)
    print("Model loaded!")

else:  # Create new model
    embedding_module = get_embedding_module(image_size=config.IMAGE_SIZE)
    siamese_net = get_siamese_network(
        image_size=config.IMAGE_SIZE, embedding_model=embedding_module
    )
    siamese_model = SiameseModel(siamese_net=siamese_net, margin=0.5)
    siamese_model.compile(optimizer=tf.keras.optimizers.SGD(config.LEARNING_RATE))


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(config.OUTPUT_PATH/"{epoch:02d}-{val_loss:.2f}.keras"),
    save_freq="epoch",
    verbose=1,
    monitor="val_loss",
    save_best_only=True
)

try:
    print("Training the siamese model...")
    siamese_model.fit(
        train_ds,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        validation_data=valid_ds,
        validation_steps=config.VALIDATION_STEPS,
        epochs=config.EPOCHS,
        callbacks=[cp_callback]
    )

except KeyboardInterrupt as e:
    print(F"Interrupted by user!")
    print(f"Saving the siamese network to {config.MODEL_PATH}...")
    config.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    tf.keras.models.save_model(
        model=siamese_model,
        filepath=config.MODEL_PATH,
        include_optimizer=True,
    )

create_index()
