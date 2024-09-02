# Standard Library imports
import pickle

# External imports
from tensorflow import keras
import numpy as np
import tensorflow as tf
import faiss
from tqdm import tqdm
import pandas as pd

# Local imports
from siamese.dataset import CommonMapFunction
import siamese.config as config


def create_one_head_net(model_path):
    """
    For creating embeddings of a single image
    """
    print(f"Loading the model from '{model_path}'")

    if not model_path.exists():
        raise Exception(f"Model doesn't exist: '{model_path}'")

    siamese_model = keras.models.load_model(filepath=model_path)
    embedding_layer = siamese_model.siamese_net.get_layer("embedding")
    input_tensor = keras.Input(name="anchor", shape=config.IMAGE_SIZE + (3,))
    embedding = embedding_layer(input_tensor)
    return keras.Model(inputs=[input_tensor], outputs=[embedding])


def create_faiss_index(model_path, data_path):
    """ """

    print("Creating Faiss index...")

    one_head_net = create_one_head_net(model_path)
    index = faiss.IndexFlatIP(config.EMBEDDING_SHAPE)
    map_fun = CommonMapFunction(config.IMAGE_SIZE)

    images_paths = list(data_path.rglob("*.jpg"))
    num_images = len(images_paths)
    images_df = save_images_df(images_paths)
    index_data = np.zeros(
        (num_images, config.EMBEDDING_SHAPE), dtype=np.float32
    )  # faiss can only work with float32

    for i, row in tqdm(images_df.iterrows()):
        image = map_fun.decode_and_resize(str(row.image_path))
        image = tf.expand_dims(image, 0, name=None)  # add batch dimension
        embedding = one_head_net(image)
        embedding = embedding.numpy()
        faiss.normalize_L2(embedding)
        index_data[i, :] = embedding

    index.add(index_data)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    print(f"Faiss index created at '{config.FAISS_INDEX_PATH}'")


def create_manual_index(model_path, data_path):
    """ """

    print("Creating dict index...")

    one_head_net = create_one_head_net(model_path)
    map_fun = CommonMapFunction(config.IMAGE_SIZE)

    images_paths = list(data_path.rglob("*.jpg"))
    num_images = len(images_paths)
    images_df = save_images_df(images_paths)
    index_data = np.zeros((num_images, config.EMBEDDING_SHAPE), dtype=np.float64)

    for i, row in tqdm(images_df.iterrows()):
        image = map_fun.decode_and_resize(str(row.image_path))
        image = tf.expand_dims(image, 0, name=None)  # add batch dimension
        embedding = one_head_net(image)
        embedding = embedding.numpy().astype(np.float64)
        embedding = embedding / np.linalg.norm(embedding)  # normalization
        index_data[i, :] = embedding

    # Save index
    with open(config.MANUAL_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)
    print(f"Dict index created at {config.MANUAL_INDEX_PATH}")


def save_images_df(images_paths: list):

    names = []
    paths = []
    for impath in images_paths:
        names.append(impath.name)
        paths.append(str(impath))

    df = pd.DataFrame({"image_name": names, "image_path": paths})

    df.to_csv(config.IMAGES_DF_PATH)

    return df


if __name__ == "__main__":
    if config.LOAD_MODEL_PATH is None:
        print("There is no model to load")
        exit(-1)

    model_path = config.LOAD_MODEL_PATH
    data_path = config.DATA_SUBSET

    if config.INDEX_TYPE == "faiss":
        create_faiss_index(model_path, data_path)
    elif config.INDEX_TYPE == "dict":
        create_manual_index(model_path, data_path)
