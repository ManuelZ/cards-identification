# Standard Library imports
import os

# External imports
import tensorflow as tf
import faiss
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Local imports
import siamese.config as config
from siamese.dataset import MapFunction
from siamese.create_index import create_one_head_net

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def display_query_results(im_query, distances, indices, nrows=2, ncols=5):
    """ """

    assert nrows*ncols == N_RESULTS+1

    fig = plt.figure(figsize=(12, 8))  # w,h
    plt.subplot(nrows, ncols, 1)
    plt.imshow(im_query)
    plt.title("Query")

    for i, (dist, im_idx) in enumerate(zip(distances, indices)):
        image_path = images_paths[im_idx]
        im_pred = cv2.imread(str(image_path))
        im_pred = cv2.cvtColor(im_pred, cv2.COLOR_BGR2RGB)
        plt.subplot(nrows, ncols, i+1+1) # one corresponds to the query image
        plt.imshow(im_pred)
        plt.title(f"{dist:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



one_head_net = create_one_head_net()
map_fun = MapFunction(config.IMAGE_SIZE)
index = faiss.read_index(str(config.INDEX_PATH))
print(f"There are {index.ntotal} observations in the index")

images_paths = list(config.DATASET.rglob("*.jpg"))

N_RESULTS = 9

for query_path in config.QUERY_DATASET.rglob("*.jpg"):
    
    # Load and preprocess image
    image = map_fun.decode_and_resize(str(query_path))
    
    # Add batch dimension
    image = tf.expand_dims(image, 0, name=None)
    
    # Extract embeddings
    embedding = one_head_net(image).numpy()
    
    # Normalize to unit vector
    faiss.normalize_L2(embedding)

    # Query the index
    distances, indices = index.search(embedding, N_RESULTS)
    indices = indices.ravel().tolist()
    distances = distances.ravel().tolist()

    # Display query image
    im_query = cv2.imread(str(query_path))
    im_query = cv2.cvtColor(im_query, cv2.COLOR_BGR2RGB)

    display_query_results(im_query, distances, indices)