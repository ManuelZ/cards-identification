# Standard Library imports
import random
import cv2

# External imports
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local imports
import siamese.config as config
from siamese.utils import get_image_paths


common_transforms = A.Compose(
    [
        A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        A.Normalize(),
        ToTensorV2(),
    ]
)


class SiameseDataset(torch.utils.data.Dataset):
    """ """

    def __init__(self, dataset, common_transforms, aug_transforms):
        self.filepaths = get_image_paths(dataset, return_str=True)
        self.total_files = len(self.filepaths)
        self.common_transforms = common_transforms
        self.aug_transforms = aug_transforms

    def __len__(self):
        return len(self.filepaths)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_negative(self, anchor_path):
        temp = self.filepaths.copy()
        temp.remove(anchor_path)
        return random.choice(temp)

    def __getitem__(self, idx):

        # Load and process the anchor
        anchor_path = self.filepaths[idx]
        anchor_numpy = self.load_image(anchor_path)
        anchor = self.common_transforms(image=anchor_numpy)["image"]

        # Create a positive by augmentation
        positive = self.aug_transforms(image=anchor_numpy)["image"]
        positive = self.common_transforms(image=positive)["image"]

        return anchor, positive
