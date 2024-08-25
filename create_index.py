# External imports
from tensorflow import keras
import tensorflow as tf
import faiss
from tqdm import tqdm

# Local imports
from siamese.dataset import MapFunction
import siamese.config as config


def create_one_head_net(model_path):
    """
    For creating embeddings of a single image
    """
    print(f"Loading the siamese network from {model_path}...")
    
    siamese_model = keras.models.load_model(filepath=model_path)
    embedding_layer = siamese_model.siamese_net.get_layer("embedding")
    input_tensor = keras.Input(name="anchor", shape=config.IMAGE_SIZE + (3,))
    embedding = embedding_layer(input_tensor)
    return keras.Model(inputs=[input_tensor], outputs=[embedding])

def create_index(model_path):
    """ """
    
    one_head_net = create_one_head_net(model_path)
    num_features = 128
    index = faiss.IndexFlatIP(num_features)
    map_fun = MapFunction(config.IMAGE_SIZE)

    for filepath in tqdm(config.DATASET.rglob("*.jpg")):
        image = map_fun.decode_and_resize(str(filepath))
        image = tf.expand_dims(image, 0, name=None)  # add batch dimension
        embedding = one_head_net(image)
        embedding = embedding.numpy()
        faiss.normalize_L2(embedding)
        index.add(embedding)

    faiss.write_index(index, str(config.INDEX_PATH))


if __name__ == "__main__":
    create_index()
