import torch
import torch.nn as nn
import torch.nn.functional as F

#todo : device!
#todo : accuracy + val loss + val acc a la fin ? ou a chaque epoch ?
#todo : mode verbose (ie a chaque epoch) ou meme auto ? genre 25% 50%

def train(model, train_loader, optimizer:torch.optim = None, criterion = None, epochs:int = 50):
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if criterion is None:
        criterion = F.cross_entropy

    for epoch in range(epochs):
        for batch_idx, (Xb, Yb) in enumerate(train_loader):
            logits = model(Xb)
            loss = criterion(logits, Yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch {}/{}. Loss={}".format(epoch, epochs, loss.item()))