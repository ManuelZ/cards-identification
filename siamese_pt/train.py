# External imports
import torch
from torch.utils.tensorboard import SummaryWriter
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
    num_workers=config.NUM_WORKERS,
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
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
    """ """
    checkpoint = torch.load(config.LOAD_MODEL_PATH_PT, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(
        f"Previous state has been loaded! Previous epoch was {epoch} with a loss of {loss:.4f}."
    )
    return model, optimizer, epoch, loss


class Trainer:
    def __init__(
        self, model, optimizer, loss_fun, best_loss, train_dataloader, valid_dataloader
    ):
        """
        best_loss: The validation loss achieved by the loaded moadel.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.best_loss = best_loss
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss_meter = AverageMeter("Loss", ":.4e")
        self.writer = SummaryWriter()

    def evaluate(self, tensor):
        tensor = tensor.to(DEVICE, dtype=torch.float32)
        embeddings = self.model(tensor)
        return embeddings

    def train(self):
        """ """

        self.loss_meter.reset()
        pbar = tqdm(self.train_dataloader)

        self.model.train()
        for anchor, positive in pbar:
            anchor_embeddings = self.evaluate(anchor)
            positive_embeddings = self.evaluate(positive)
            loss = self.loss_fun(anchor_embeddings, positive_embeddings)
            self.loss_meter.update(loss.item(), anchor.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_description(f"Batch loss={loss.item():.5f}")

        return self.loss_meter.avg

    def test(self):
        """ """

        self.loss_meter.reset()
        pbar = tqdm(self.valid_dataloader)

        self.model.eval()
        with torch.no_grad():
            for anchor, positive in pbar:
                anchor_embeddings = self.evaluate(anchor)
                positive_embeddings = self.evaluate(positive)
                loss = self.loss_fun(anchor_embeddings, positive_embeddings)
                self.loss_meter.update(loss.item(), anchor.size(0))

                pbar.set_description(f"Batch loss={loss.item():.5f}")

        return self.loss_meter.avg

    def run(self):
        """ """

        for epoch in range(starting_epoch, config.EPOCHS + 1):
            print(f"Starting epoch {epoch}")

            train_loss = trainer.train()
            test_loss = trainer.test()

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", test_loss, epoch)

            print(
                f"Epoch {epoch+1:} | train loss: {train_loss:.6f}; valid loss: {test_loss:.6f} "
            )

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                save_state(self.model, self.optimizer, epoch, test_loss)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    From: https://github.com/pytorch/examples/blob/3970e068c7f18d2d54db2afee6ddd81ef3f93c24/imagenet/main.py#L363
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":

    model = create_model()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM
    )
    loss_fun = SelfSupervisedLoss(CircleLoss(m=0.25, gamma=256))

    starting_epoch = 1
    best_loss = float("inf")

    if config.LOAD_MODEL_PATH_PT.exists():
        model, optimizer, starting_epoch, best_loss = load_state(model, optimizer)
        print(f"Restarting training. Previous validation loss: {best_loss:.4f}")

    trainer = Trainer(model, optimizer, loss_fun, best_loss, train_loader, valid_loader)
    trainer.run()
    print("Finished")
