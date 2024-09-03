# External imports
import torch
from pytorch_metric_learning.losses import (
    SelfSupervisedLoss,
    TripletMarginLoss,
    CircleLoss,
)
from tqdm import tqdm

# Local imports
from siamese.siamese_pt.dataset import SiameseDataset, common_transforms
from siamese.siamese_pt.model import create_model
from siamese.augmentations import al_augmentations
import siamese.config as config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = SiameseDataset(
    config.TRAIN_DATASET,
    common_transforms=common_transforms,
    aug_transforms=al_augmentations,
)

valid_dataset = SiameseDataset(
    config.VALID_DATASET,
    common_transforms=common_transforms,
    aug_transforms=al_augmentations,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=0,
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=0,
)

lr = 1e-4
momentum = 0.937
num_epochs = 10

model = create_model()
loss_func = SelfSupervisedLoss(CircleLoss(m=0.25, gamma=256))
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)


def train(dataloader):
    epoch_loss = 0
    num_steps = len(dataloader.dataset) // config.BATCH_SIZE

    model.train()
    pbar = tqdm(dataloader)
    for anchor, positive in pbar:
        anchor = anchor.to(DEVICE, dtype=torch.float32)
        positive = positive.to(DEVICE, dtype=torch.float32)
        anchor_embeddings = model(anchor)
        positive_embeddings = model(positive)
        loss = loss_func(anchor_embeddings, positive_embeddings)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss={loss.item():.5f}")

    avg_epoch_loss = epoch_loss / num_steps
    print(f"Epoch train loss: {avg_epoch_loss:.6f}")


def test(dataloader):
    epoch_loss = 0
    num_steps = len(dataloader.dataset) // config.BATCH_SIZE

    model.eval()
    pbar = tqdm(dataloader)
    for anchor, positive in pbar:
        anchor = anchor.to(DEVICE, dtype=torch.float32)
        positive = positive.to(DEVICE, dtype=torch.float32)
        anchor_embeddings = model(anchor)
        positive_embeddings = model(positive)
        loss = loss_func(anchor_embeddings, positive_embeddings)
        epoch_loss += loss.item()

        pbar.set_description(f"loss={loss.item():.5f}")

    avg_epoch_loss = epoch_loss / num_steps
    print(f"Epoch valid loss: {avg_epoch_loss:.6f} ")


for epoch in range(1, num_epochs + 1):
    print(f"Starting epoch {epoch}")
    train(train_loader)
    test(valid_loader)

torch.save(model.state_dict(), "densenet121_model.pth")

print("Finished")
