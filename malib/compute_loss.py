import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss(model, train_loader_unshuffled, criterion = None, num_batches:int = 8):

    if criterion is None:
        criterion = F.cross_entropy

    loss = 0

    for batch_idx, (Xb, Yb) in enumerate(train_loader_unshuffled):
        logits = model(Xb)
        loss += criterion(logits, Yb).item()

        if batch_idx + 1 >= num_batches:
            break
    
    loss = loss / (batch_idx + 1)

    return loss