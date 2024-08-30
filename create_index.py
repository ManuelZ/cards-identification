# Standard Library imports
import pickle

# External imports
from tensorflow import keras
import numpy as np
import tensorflow as tf
import faiss
from tqdm import tqdm

# Local imports
from siamese.dataset import CommonMapFunction
import siamese.config as config


def create_one_head_net(model_path):
    """
    For creating embeddings of a single image
    """
    print(f"Loading the siamese network from {model_path}...")

    if not model_path.exists():
        raise Exception("Model doesn't exist: '{model_path}'")

    siamese_model = keras.models.load_model(filepath=model_path)
    embedding_layer = siamese_model.siamese_net.get_layer("embedding")
    input_tensor = keras.Input(name="anchor", shape=config.IMAGE_SIZE + (3,))
    embedding = embedding_layer(input_tensor)
    return keras.Model(inputs=[input_tensor], outputs=[embedding])


def create_faiss_index(model_path, data_path):
    """ """

    one_head_net = create_one_head_net(model_path)
    index = faiss.IndexFlatIP(config.EMBEDDING_SHAPE)
    map_fun = CommonMapFunction(config.IMAGE_SIZE)

    for filepath in tqdm(data_path.rglob("*.jpg")):
        image = map_fun.decode_and_resize(str(filepath))
        image = tf.expand_dims(image, 0, name=None)  # add batch dimension
        embedding = one_head_net(image)
        embedding = embedding.numpy()
        faiss.normalize_L2(embedding)
        index.add(embedding)

    faiss.write_index(index, str(config.FAISS_INDEX_PATH))


def create_manual_index(model_path, data_path):
    """ """

    one_head_net = create_one_head_net(model_path)
    map_fun = CommonMapFunction(config.IMAGE_SIZE)

    images_paths = list(data_path.rglob("*.jpg"))
    num_images = len(images_paths)
    
    index = np.zeros((num_images, config.EMBEDDING_SHAPE), dtype=np.float64)
    for i, filepath in tqdm(enumerate(images_paths)):
        image = map_fun.decode_and_resize(str(filepath))
        image = tf.expand_dims(image, 0, name=None)  # add batch dimension
        embedding = one_head_net(image)
        embedding = embedding.numpy().astype(np.float64)
        embedding = embedding / np.linalg.norm(embedding)  # normalization
        index[i, :] = embedding

    # Save index
    with open(config.MANUAL_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)


if __name__ == "__main__":
    model_path = config.LOAD_MODEL_PATH
    data_path = config.DATASET_SUBSET

    create_faiss_index(model_path, data_path)
    # create_manual_index(model_path, data_path)

    # Read index
    with open(config.MANUAL_INDEX_PATH, "rb") as f:
        index = pickle.load(f)
