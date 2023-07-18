import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def _get_weights(model):
    return [p.data for p in model.parameters()]

def _get_random_weights(weights):
    return [torch.randn(w.size()).to(device) for w in weights]

def create_random_directions(model):
    """
    Return two random directions in the model's weights space.
    These vectors are normalized according to https://arxiv.org/abs/1712.09913.

    Parameters
    ----------
    model : the torch model whose weights will be used to create and normalize the directions.

    Returns
    ----------
    directions : list of two tensors, which correspond to the two sampled directions.


    Notes
    ----------
    Inspired from https://github.com/tomgoldstein/loss-landscape.

    """

    x_direction = create_random_direction(model)
    y_direction = create_random_direction(model)

    return [x_direction, y_direction]

def create_random_direction(model):
    """
    Return a random direction in the model's weights space.
    This vector is normalized according to https://arxiv.org/abs/1712.09913.

    Parameters
    ----------
    model : the torch model whose weights will be used to create and normalize the direction

    Returns
    ----------
    direction : a tensor, which correspond to the sampled direction.


    Notes
    ----------
    Inspired from https://github.com/tomgoldstein/loss-landscape.

    """

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
