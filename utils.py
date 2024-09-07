# Standard Library imports
from pathlib import Path

# External imports
import torch
import pandas as pd

# Local imports
import config as config


def get_image_paths(folder: Path, return_str=False) -> list[Path | str]:
    """Get all the image paths from a folder."""

    paths = []
    for ext in config.EXTENSIONS:
        folder_paths = list(folder.rglob(ext))
        if return_str:
            folder_paths = [str(f) for f in folder_paths]
        paths.extend(folder_paths)
    return paths


def denormalize(tensors):
    """
    Denormalize image tensors back to range [0.0, 1.0]

    Modified from: Deep Learning with PyTorch - OpenCV University
    """

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    tensors = tensors.clone()
    for c in range(3):
        if len(tensors.shape) == 4:
            tensors[:, c, :, :].mul_(std[c]).add_(mean[c])
        elif len(tensors.shape) == 3:
            tensors[c, :, :].mul_(std[c]).add_(mean[c])
        else:
            raise Exception(
                "Can only deal with images of shape (N, C, H, W) or (C, H, W)"
            )

    return torch.clamp(tensors.cpu(), 0.0, 1.0)


def torch_to_cv2(image):
    """Convert a PyTorch image tensor to an OpenCV image."""

    if image.ndim == 4:
        image = image.squeeze()

    return image.permute(1, 2, 0).cpu().numpy()


def save_images_df(images_paths: list):
    """ """

    names = []
    paths = []
    for impath in images_paths:
        names.append(impath.name)
        paths.append(str(impath))

    df = pd.DataFrame({"image_name": names, "image_path": paths})

    df.index.rename("label", inplace=True)

    df.to_csv(config.IMAGES_DF_PATH)

    return df
