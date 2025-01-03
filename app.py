# Standard Libary imports
from io import BytesIO

# External imports
import gradio as gr
import torch
import cv2
import faiss
import pandas as pd
import requests
from PIL import Image
import numpy as np

# Local imports
from siamese_pt.model import create_model
from siamese_pt.dataset import common_transforms
from test_index import read_index, query_index
import config


"""
Usage:
    gradio run.py --demo-name=my_demo.

Calling with `gradio` instead of with `python` enables auto-reload mode automatically.
Gradio specifically looks for a Gradio Blocks/Interface demo called "demo" in the code.

https://www.gradio.app/guides/developing-faster-with-reload-mode
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model():
    """Creates and loads the model with pre-trained weights."""
    model = create_model()
    checkpoint = torch.load(config.LOAD_MODEL_PATH_PT, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_resources():
    """Loads the index and dataframes for image and oracle information."""
    index = read_index()
    image_df = pd.read_csv(config.IMAGES_DF_PATH)
    oracle_df = pd.read_csv(config.ROOT / "oracle-cards-20240821210300.csv")
    return index, image_df, oracle_df


def handle_image_upload(image):

    if image is None:
        return None, "No photo uploaded", "", ""

    # Prepare image before passing to model
    image = common_transforms(image=image)["image"]
    image = image.to(DEVICE, dtype=torch.float32)
    image = image.unsqueeze(0)

    # Create embeddings
    embedding = model(image)
    embedding = embedding.detach().cpu().numpy()

    # Query the index for the best match
    faiss.normalize_L2(embedding)
    indices, distances = query_index(embedding, index, config.INDEX_TYPE, n_results=4)
    best_match_distance, best_match_index = distances[0], indices[0]

    # Load match image and data
    image_name = image_df.loc[best_match_index].image_name
    image_path = image_df.loc[best_match_index].image_path
    im_pred = cv2.imread(str(image_path))
    im_pred = cv2.cvtColor(im_pred, cv2.COLOR_BGR2RGB)
    im_id = image_name.split(".")[0]
    oracle_result = oracle_df.loc[oracle_df.id == im_id]

    return (
        im_pred,
        f"{oracle_result.name.values[0]}",
        f"{best_match_distance:.3f}",
        f"{oracle_result.scryfall_uri.values[0]}",
    )


def download_image_as_numpy(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Open the image and convert to a NumPy array
        image = Image.open(BytesIO(response.content))
        image_np = np.array(image)
        return image_np
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
        return None


def show_image(df, event_data: gr.EventData):
    """
    evt.target contains <gradio.components.dataframe.Dataframe object at 0x000001B107557910>
    """
    name, image_uri = event_data._data["row_value"]
    image = download_image_as_numpy(image_uri)
    return name, image


image_upload_tab_text = """
    ## Welcome to the *Magic: The Gathering* Card Matching App!

    Easily identify Magic: The Gathering cards by uploading an image. The app finds the best match from a curated dataset.

    **Note: The dataset includes only cards without reprints.**
    Some cards have been reprinted with new artwork, but only cards without reprints have been indexed.
    A card may not be correctly identified if it has not been included in the index.
    Use the **Indexed Cards** tab to browse the included cards.
    
    ### How to Use
    1. Use the **Image Upload** tab to upload an image and view the closest match.
    2. Use the **Indexed Cards** tab to to explore the included cards in the dataset. 
    
    To test the app, consider searching for card images in platforms like eBay.
    """

table_tab_text = """
    The table below lists all the images included in the database. Cards not shown here are likely excluded due to having multiple reprints.
"""


def create_ui(oracle_df):
    """
    Create a Gradio UI.
    """

    with gr.Blocks() as demo:

        with gr.Tab("Image Upload"):

            with gr.Row():

                with gr.Column():
                    gr.Markdown(image_upload_tab_text)
                    image_input = gr.Image(type="numpy")

                with gr.Column():
                    image_input.change(
                        handle_image_upload,
                        inputs=image_input,
                        outputs=[
                            gr.Image(label="Uploaded Photo"),
                            gr.Textbox(label="Card name"),
                            gr.Textbox(label="Confidence"),
                            gr.Textbox(label="URI"),
                        ],
                    )

        with gr.Tab("Indexed Cards"):
            with gr.Row():
                with gr.Column():

                    gr.Markdown(table_tab_text)

                    # Table for row selection
                    table = gr.Dataframe(
                        headers=["Card Name", "URI"],
                        datatype=["str", "str"],
                        value=oracle_df.loc[:, ["name", "image_uri"]],
                        interactive=True,
                    )

                with gr.Column():
                    # Output for selected row
                    selected_info = gr.Textbox(
                        label="Selected image", interactive=False
                    )
                    image_output = gr.Image(type="numpy")

            # Link table row selection to outputs
            table.select(
                show_image, inputs=[table], outputs=[selected_info, image_output]
            )

    return demo


if __name__ == "__main__":
    model = initialize_model()
    index, image_df, oracle_df = load_resources()
    demo = create_ui(oracle_df)
    demo.launch()
