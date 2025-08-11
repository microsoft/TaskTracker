import os

import torch
from sklearn.metrics import roc_auc_score


def save_checkpoint(state, save_path, filename):
    """Save the training checkpoint.

    Parameters:
        state (dict): State to be saved, typically includes model and optimizer state dicts.
        save_path (str): Directory where the checkpoint will be saved.
        filename (str): Name of the checkpoint file.
    """
    torch.save(state, os.path.join(save_path, filename))


def load_checkpoint(checkpoint_path, model, optimizer):
    """Load a training checkpoint and update model and optimizer states.

    Parameters:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model instance to update state dict.
        optimizer (torch.optim.Optimizer): Optimizer instance to update state dict.

    Returns:
        tuple: Returns the epoch and best validation loss from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("best_roc_auc", 0)


def compute_ROC_AUC(distances_clean, distances_poisoned):
    max_dist = max(max(distances_clean), max(distances_poisoned))
    distances_clean = [(item / max_dist) for item in distances_clean]
    distances_poisoned = [(item / max_dist) for item in distances_poisoned]
    all_distances = distances_clean + distances_poisoned
    labels = [0 for i in range(0, len(distances_clean))] + [
        1 for i in range(0, len(distances_poisoned))
    ]
    roc_auc = roc_auc_score(labels, all_distances)
    return roc_auc
