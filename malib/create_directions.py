import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def _get_weights(model):
    return [p.data for p in model.parameters()]

def _get_random_weights(weights):
    return [torch.randn(w.size()).to(device) for w in weights]

def create_random_directions(model):
    x_direction = create_random_direction(model)
    y_direction = create_random_direction(model)

    return [x_direction, y_direction]

def create_random_direction(model):
    weights = _get_weights(model)
    direction = _get_random_weights(weights)
    _normalize_directions_for_weights(direction, weights)

    return direction

def _normalize_directions_for_weights(direction, weights):
    assert (len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            d.fill_(0) 
        d.mul_(w.norm() / (d.norm() + 1e-10))
