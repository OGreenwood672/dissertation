import torch
from numpy import sqrt


def init_layer(layer, std=sqrt(2), bias_const=0.0):
    """
    Helper function to initialise weights for PPO layers.
    Common method for initialising RL weights
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer