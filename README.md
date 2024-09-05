# Card recognition

Given a query card and a database of 30K cards, the goal is to determine which card in the database matches the query card.

![image](https://github.com/user-attachments/assets/f7821999-0585-45d2-888e-9f032496e599)


## Introduction

This experiment is inspired in the problem of face recognition, which tries to answer the question: "Who is this person?".

## The dataset

A total of 30467 unique cards from the game *Magic: The gathering* were downloaded from [Scryfall](https://scryfall.com/docs/api/bulk-data).

While there are reprinted card editions with different artwork, only one version of each card is used in this experiment.

## The method: overview

Fine-tune a feature extractor network using a similarity loss function: Circle Loss. Once the network is trained, use it to extract embeddings for the 30467 cards in the dataset and create an index that's used to compare the stored representation vector of each card with the one of the query card.

The training aims to make the network learn an embedding space where the embeddings of images of the same class will be appear close to each other and the embeddings of images of different classes will be far apart from each other.

This learning scheme requires feeding the loss function with examples of cards from the same class as well as cards from different classes.

Since the dataset contains only one image per card, how can the network learn to generate embeddings that are close in space if it only sees one observation of each card? The solution is to provide augmented versions of each card, allowing the network to compare the original with the augmented version and treat them as different instances of the same class.

Additionally, the learning process involves using examples of cards from different classes, so the network learns to distance their embeddings from one another.

## The magic inside the PyTorch Metric Learning library

The loss function used comes from the "PyTorch Metric Learning" library, which makes it easy to apply a pairwise loss function to a batch of images. Following the assumption that each card is unique (i.e., there is only one observation per class), the loss function also handles creating all possible pairs or triplets from a batch to feed them to its core logic. An interesting aspect is how the data is fed: pairs are not provided one by one.

- The `SelfSupervisedLoss` class is used to wrap the loss function, so that it can consume two tensors, one of images and one of augmented images. Otherwise, it would have to receive tensors and labels. The idea here is that there is only one image per class, so the process of assigning labels can be automated internally without risk.

- The [`CircleLoss` class](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/28d5acce05f61b0d0c15acbe423a7ba387efcb4d/src/pytorch_metric_learning/losses/circle_loss.py#L10) has an internal method `_compute_loss` where the logic of the [paper](https://arxiv.org/pdf/2002.10857) is applied (notice the use of `logsumexp` and `soft_plus`). It also has access, through inheritance, to some important methods: `.mat_based_loss` `.compute_loss` and `.forward`. This class descends from `torch.nn.Module`, that's why it depends on a `.forward` method, which gets called when an instance of `CircleLoss` is called.

    - `.forward(embeddings, labels)` (part of `BaseMetricLossFunction`) calls...
    - `.compute_loss(embeddings, labels, ...)` (part of `GenericPairLoss`) calls...
    - `.mat_based_loss()` (part of `GenericPairLoss`) calls...
    - `._compute_loss` (part of `CircleLoss`)

- Part of the inheritance hierarchy looks like this:

    BaseMetricLossFunction -> GenericPairLoss -> CircleLoss

- In `compute_loss` and `mat_based_loss`, a list of embeddings and auto-generated labels get converted to:
    - A matrix storing Cosine distances between all embeddings
    - Two binary matrices of indexes that allow to identify:
        - which elements in the matrix correspond to distances between anchor-positive pairs
        - which elements in the matrix correspond to distances between anchor-negative pairs
- In `_compute_loss`, part of `CircleLoss`, the Circle Loss logic is applied for all the combinations of anchor-positive and anchor-negative pairs that can be obtained from the original batch of images, expressed in the 3 matrices mentioned above.


## How to use the results

Once the neural network has been fine-tuned using the Circle Loss, it should have learned how to generate embeddings of images so that they cluster together for images belonging to the same class. It is possible now to generate embeddings of a dataset, store them and compare them later as needed (through the Cosine Distance), with the embeddings of query images, and determine if they are to be considered part of the same class.

This comparison is currently done using the Faiss library, but I have not tested any search method besides the naive one of comparing the query with all the stored elements.
