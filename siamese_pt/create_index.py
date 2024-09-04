# External imports
import cv2
import numpy as np
import torch
import faiss
from tqdm import tqdm

# Local imports
from siamese.siamese_pt.model import create_model
from siamese.siamese_pt.dataset import SiameseDataset, common_transforms
from siamese.augmentations import al_augmentations
from siamese.utils import get_image_paths, save_images_df
import siamese.config as config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = SiameseDataset(
    config.DATA,
    common_transforms=common_transforms,
    aug_transforms=al_augmentations,
)


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    num_workers=0,
)


def create_faiss_index(model, data_path, index_path):
    """ """

    print("Creating Faiss index...")

    num_images = len(test_loader)
    index_data = np.zeros(
        (num_images, config.EMBEDDING_SHAPE), dtype=np.float32
    )  # faiss can only work with float32
    index = faiss.IndexFlatIP(config.EMBEDDING_SHAPE)

    images_paths = get_image_paths(data_path, return_str=False)
    num_images = len(images_paths)
    images_df = save_images_df(images_paths)
    index_data = np.zeros(
        (num_images, config.EMBEDDING_SHAPE), dtype=np.float32
    )  # faiss can only work with float32

    for i, row in tqdm(images_df.iterrows()):
        image = cv2.imread(str(row.image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = common_transforms(image=image)["image"]
        image = image.to(DEVICE, dtype=torch.float32)
        image = image.unsqueeze(0)
        embedding = model(image)
        embedding = embedding.detach().cpu().numpy()
        faiss.normalize_L2(embedding)
        index_data[i, :] = embedding

    index.add(index_data)
    faiss.write_index(index, str(index_path))
    print(f"Faiss index created at '{index_path}'")


if __name__ == "__main__":
    model = create_model()
    model.load_state_dict(torch.load("densenet121_model.pth"))
    model.eval()

    create_faiss_index(model)
