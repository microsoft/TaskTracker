import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, step=None):
        # Calculate pairwise distances
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)

        # Compute loss based on the margin
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


def triplet_mining(anchor, positive, negative, margin, hard=False):
    indices = []

    # distance between anchor and positive per row (each example)
    distance_ap_per_row = F.pairwise_distance(anchor, positive, 2)

    # distance between each anchor and all negatives
    # should be a square matrix of batch_size x batch_size
    # distance_an_square[i,j] is the distance between anchor i and negative j
    distance_an_square = []
    for i in range(anchor.size(0)):
        distance_anchor_i_all_neg = F.pairwise_distance(
            anchor[i].repeat(negative.size(0), 1), negative, 2
        )
        distance_an_square.append(distance_anchor_i_all_neg)
    distance_an_square = torch.stack(distance_an_square, dim=0)

    # get semi hard negatives for each anchor, positive pairs
    for i in range(anchor.size(0)):
        # compute loss between each (anchor, positive) and all negatives
        loss_i = F.relu(
            distance_ap_per_row[i].repeat(
                negative.size(0),
            )
            - distance_an_square[i, :]
            + margin
        )
        # semi hard condition. Loss is larger than 0 but less than the margin
        indices_neg = torch.where((loss_i > 0) & (loss_i < margin))[0]

        # for triplet combinations. First index is (anchor, positive). Second index is negatives
        for index_neg in indices_neg:
            indices.append(torch.tensor([i, index_neg.item()]))

        # Hard condition. Loss is larger than margin
        if hard:
            indices_neg = torch.where(loss_i > margin)[0]
            for index_neg in indices_neg:
                indices.append(torch.tensor([i, index_neg.item()]))

    # stack and shuffle indices
    if len(indices) != 0:
        indices = torch.stack(indices, dim=0)
        indices = indices[torch.randperm(indices.size(0)), :]

    return indices


def triplet_mining_unique(
    anchor, positive, negative, margin, text_batch, hard=False, step=None
):
    """Updated triplet mining but first removing duplicates of pairs of (primary, clean) in each mini batch"""
    unique_primary_clean = find_unique_indices(text_batch)
    print(len(unique_primary_clean))
    indices = []

    # Pre-calculate distances between all anchor-positive and anchor-negative pairs
    distance_ap_per_row = F.pairwise_distance(anchor, positive, 2)

    # Pre-calculate distances between each anchor and all negatives
    distance_an_square = []
    for i in range(anchor.size(0)):
        # distance_anchor_i_all_neg = F.pairwise_distance(anchor[i].unsqueeze(0).expand_as(negative), negative, 2)
        distance_anchor_i_all_neg = F.pairwise_distance(
            anchor[i].repeat(negative.size(0), 1), negative, 2
        )
        distance_an_square.append(distance_anchor_i_all_neg)
    distance_an_square = torch.stack(distance_an_square, dim=0)

    num_semi_hard_negatives, num_hard_negatives = (
        0,
        0,
    )  # Initialize counters for MLflow logging

    # Identify semi-hard and hard negatives based on the specified margin
    for i in range(anchor.size(0)):
        if not i in unique_primary_clean:
            continue
        loss_i = F.relu(
            distance_ap_per_row[i].repeat(
                negative.size(0),
            )
            - distance_an_square[i, :]
            + margin
        )

        indices_neg_semi_hard = torch.where((loss_i > 0) & (loss_i < margin))[0]
        num_semi_hard_negatives += len(
            indices_neg_semi_hard
        )  # Count semi-hard negatives

        indices_neg_hard = torch.tensor([], dtype=torch.long)
        if hard:
            indices_neg_hard = torch.where(loss_i > margin)[0]
            num_hard_negatives += len(indices_neg_hard)  # Count hard negatives

        for index_neg in indices_neg_semi_hard:
            indices.append(torch.tensor([i, index_neg.item()]))
        if hard:
            for index_neg in indices_neg_hard:
                indices.append(torch.tensor([i, index_neg.item()]))

    if len(indices) != 0:
        indices = torch.stack(indices, dim=0)
        indices = indices[torch.randperm(indices.size(0)), :]

    return indices


def find_unique_indices(text_batch):
    unique = set()
    indices = set()
    for i in range(0, len(text_batch)):
        if text_batch[i] in unique:
            continue
        unique.add(text_batch[i])
        indices.add(i)
    return indices
