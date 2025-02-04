import torch
import torch.nn as nn
import torch.nn.functional as F 

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()  # More explicit for compatibility

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))  # Mish activation

    def __repr__(self):
        return "Mish()"
