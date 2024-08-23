import backend.siamese.config as config
from tensorflow import keras


print(f"Loading the siamese network from {config.MODEL_PATH}...")
siamese_net = keras.models.load_model(filepath=config.MODEL_PATH)

# Create embeddings of a single image
input = keras.Input(name="anchor", shape=config.IMAGE_SIZE + (3,))
embedding = siamese_net(input)
