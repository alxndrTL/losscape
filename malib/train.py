import torch
import torch.nn as nn
import torch.nn.functional as F

#todo : accuracy + val loss + val acc a la fin ? ou a chaque epoch ?

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_loader, optimizer:torch.optim = None, criterion = None, epochs:int = 50, verbose:int = 1):
    #verbose = 0 : print seulement a la fin
    #verbose = 1 : print 0%, 25%, 50%, 100%
    #verbose = 2 : print a chaque epoch

    model.to(device)
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if criterion is None:
        criterion = F.cross_entropy

    for epoch in range(1, epochs+1):
        for batch_idx, (Xb, Yb) in enumerate(train_loader):
            Xb, Yb = Xb.to(device), Yb.to(device)

            logits = model(Xb)
            loss = criterion(logits, Yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose == 2:
            print("Epoch {}/{}. Loss={}".format(epoch, epochs, loss.item()))

        if verbose == 1 and (epoch%(epochs/4) == 0):
            print("Epoch {}/{}. Loss={}".format(epoch, epochs, loss.item()))

    if verbose == 0:
        print("Epoch {}/{}. Loss={}".format(epoch, epochs, loss.item()))