import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_loss(model, train_loader_unshuffled, criterion = F.cross_entropy, num_batches:int = 8):
    """
    Compute and return the loss over the first num_batches batches given by the train_loader_unshuffled, using the criterion provided.

    Parameters
    ----------
    model : the torch model which will be evaluated.
    train_loader_unshuffled : the torch dataloader. It is supposed to be fixed so that all the calls to this function will use the same data.
    criterion : the criterion used to compute the loss. (default to F.cross_entropy)
    num_batches : number of batches to evaluate the model with. (default to 8)

    Returns
    ----------
    loss : loss computed

    """

    if criterion is None:
        criterion = F.cross_entropy

    loss = 0

    with torch.no_grad():
        for batch_idx, (Xb, Yb) in enumerate(train_loader_unshuffled):
            Xb, Yb = Xb.to(device), Yb.to(device)

            logits = model(Xb)
            loss += criterion(logits, Yb).item()

            if batch_idx + 1 >= num_batches:
                break
    
    loss = loss / (batch_idx + 1)

    return loss

#simple, light weight and modular neural newtork loss landscape viz lib

# adv:
# -1D plot
# -torch.no_grad
# -easy plug and play