# Standard Library imports
import os
import pickle

# External imports
import faiss
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
import siamese.config as config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def display_query_results(im_query, distances, indices, nrows=2, ncols=5):
    """ """

    max_images = int(nrows * ncols)

    df = pd.read_csv(config.IMAGES_DF_PATH)

    fig = plt.figure(figsize=(12, 8))  # w,h
    plt.subplot(nrows, ncols, 1)
    plt.imshow(im_query)
    plt.axis("off")
    plt.title("Query")

    for i, (dist, im_idx) in enumerate(zip(distances, indices)):

        if i == max_images:
            break

        image_path = df.loc[im_idx].image_path
        im_pred = cv2.imread(str(image_path))
        im_pred = cv2.cvtColor(im_pred, cv2.COLOR_BGR2RGB)

        plt.subplot(nrows, ncols, i + 1 + 1)  # one corresponds to the query image
        plt.imshow(im_pred)
        plt.title(f"{dist:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def query_index(embedding, index, index_type, n_results):
    """ """

    if index_type == "faiss":
        faiss.normalize_L2(embedding)
        distances, indices = index.search(embedding, n_results)
        indices = indices.ravel().tolist()
        distances = distances.ravel().tolist()

    elif index_type == "dict":
        embedding = embedding / np.linalg.norm(embedding)
        # similarities = cosine_similarity(index, embedding).ravel()

        distances = []
        for i in range(len(index)):
            d = np.linalg.norm(index[i, :] - embedding)
            distances.append(d)

        distances = np.array(distances)
        indices = distances.argsort()[:n_results]
        distances = distances[indices]

    return indices, distances


def read_index():
    """ """

    if config.INDEX_TYPE == "dict":
        with open(config.MANUAL_INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        print(f"There are {len(index)} observations in the index")

    elif config.INDEX_TYPE == "faiss":
        index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        print(f"There are {index.ntotal} observations in the index")

    return index
