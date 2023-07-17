import torch

import numpy as np
import matplotlib.pyplot as plt

from malib.compute_loss import compute_loss
from malib.create_directions import create_random_direction, create_random_directions

#todo : device!
#todo : pca!
#todo : xmin, xmax res etc etc
#todo : possibilité de faire sur le test loss aussi! et sur le même graphiqueuu
#todo : save grahpique avec nom ? où ?

#todo pour la lib : possiblité de tout foutre dans un fichier, et il fait les exps automatiquement ? (genre on met model + dataloader + optim + loss et il loop sur les models + optims)

def create_1D_losscape(model, train_loader_unshuffled, direction=None, criterion = None, num_batches:int = 8, save_only:bool = False):

    if direction is None:
        direction = create_random_direction(model)

    init_weights = [p.data for p in model.parameters()]

    coords = np.linspace(-1, 1, 10)
    losses = []

    for x in coords:
        set_weights(model, init_weights, direction, x)

        loss = compute_loss(model, train_loader_unshuffled, criterion, num_batches)
        losses.append(loss)

    reset_weights(model, init_weights)
    
    plt.plot(coords, losses)

def create_2D_losscape():
    #xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    #coords = np.c_[xx.ravel(), yy.ravel()] # (10*10, 2)

    return NotImplementedError

def set_weights(model, weights, directions, step):
    if len(directions) == 2:
        dx = directions[0]
        dy = directions[1]
        changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]

    else:
        changes = [d*step for d in directions]

    for (p, w, d) in zip(model.parameters(), weights, changes):
        p.data = w + torch.Tensor(d).type(type(w))

def reset_weights(model, weights):
    for (p, w) in zip(model.parameters(), weights):
        p.data.copy_(w.type(type(p.data)))