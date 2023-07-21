import torch
import torch.nn as nn
import torch.nn.functional as F

#todo : accuracy + val loss + val acc a la fin ? ou a chaque epoch ?

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_loader, optimizer:torch.optim = None, criterion = F.cross_entropy, epochs:int = 50, decay_lr_epochs:int = 20, verbose:int = 1):
    """
    Train the provided model.

    Parameters
    ----------
    model : the torch model which will be trained.
    train_loader : the torch dataloader which gives training data.
    optimizer : the optimizer used for training (should follow the same API as torch optimizers).(default to Adam)
    criterion : the criterion used to compute the loss. (default to F.cross_entropy)
    epochs : the number of epochs (default to 50)
    decay_lr_epochs : the lr will be divided by 10 every decay_lr_epochs epochs (default to 20)
    verbose : controls the printing during the training. (0 = print at the end only, 1 = print at 0,25,50,100%, 2 = print every epoch). (default to 1)

    Returns
    ----------

    """

    model.to(device)

    if criterion is None:
        criterion = F.cross_entropy
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
    lr = optimizer.param_groups[0]['lr']

    for epoch in range(1, epochs+1):
        for batch_idx, (Xb, Yb) in enumerate(train_loader):
            Xb, Yb = Xb.to(device), Yb.to(device)

            logits = model(Xb)
            loss = criterion(logits, Yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch%decay_lr_epochs == 0:
            lr = 0.1 * lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if verbose == 2:
            print("Epoch {}/{}. Loss={}".format(epoch, epochs, loss.item()))

        if verbose == 1 and (epoch%(epochs/4) == 0):
            print("Epoch {}/{}. Loss={}".format(epoch, epochs, loss.item()))

    if verbose == 0:
        print("Epoch {}/{}. Loss={}".format(epoch, epochs, loss.item()))