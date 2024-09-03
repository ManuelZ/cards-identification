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


def save_state(net, optimizer, epoch, loss):
    """ """

    print(f"Saving model during epoch {epoch} with a loss of {loss:.4f}")

    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        config.LOAD_MODEL_PATH_PT,
    )


def load_state(model, optimizer):
    checkpoint = torch.load(config.LOAD_MODEL_PATH_PT, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Loaded state! Previous epoch was {epoch} with a loss of {loss:.4f}.")
    return model, optimizer, epoch, loss


def train(model, dataloader):
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
    return avg_epoch_loss


def test(model, dataloader):
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
    return avg_epoch_loss


if __name__ == "__main__":

    starting_epoch = 1
    model = create_model()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM
    )
    loss_func = SelfSupervisedLoss(CircleLoss(m=0.25, gamma=256))

    if config.LOAD_MODEL_PATH_PT.exists():
        model, optimizer, starting_epoch, loss = load_state(model, optimizer)
        print(f"Restarting training. Previous validation loss: {loss:.4f}")

    best_loss = float("inf")
    for epoch in range(starting_epoch, config.EPOCHS + 1):
        print(f"Starting epoch {epoch}")
        train_loss = train(model, train_loader)
        test_loss = test(model, valid_loader)
        if test_loss < best_loss:
            best_loss = test_loss
            save_state(model, optimizer, epoch, test_loss)

    print("Finished")
