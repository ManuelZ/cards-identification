# External imports
from tensorflow import keras
import tensorflow as tf

# Local imports
from siamese.dataset import PairsGenerator
from siamese.dataset import MapFunction
import siamese.config as config


print(f"Loading the siamese network from {config.MODEL_PATH}...")
siamese_net = keras.models.load_model(filepath=config.MODEL_PATH)

# For creating embeddings of a single image
network_input = keras.Input(name="anchor", shape=config.IMAGE_SIZE + (3,))
embedding = siamese_net(network_input)
one_head_net = keras.Model(inputs=[network_input], outputs=[embedding])

# Create a TF dataset
data_generator = PairsGenerator(datasetPath=config.VALID_DATASET)
dataset = tf.data.Dataset.from_generator(
    generator=data_generator.get_next_element,
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
    ),
)
map_fun = MapFunction(image_size=config.IMAGE_SIZE)
dataset = dataset.map(map_fun).batch(4).prefetch(config.AUTO)
(anchor, positive, negative) = next(iter(dataset))
anchor_embedding = one_head_net(anchor)
