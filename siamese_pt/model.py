# External imports
import torch
import torchvision

# Local imports
import config as config


def create_model():

    model = torchvision.models.densenet121(
        weights=torchvision.models.DenseNet121_Weights.DEFAULT,
        memory_efficient=True,
    )

    model.classifier = torch.nn.Linear(
        model.classifier.in_features, config.EMBEDDING_SHAPE
    )
    if torch.cuda.is_available():
        model.cuda()
    return model
