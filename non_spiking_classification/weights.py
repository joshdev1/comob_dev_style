import torch
import torch.nn as nn
import numpy as np

from parameters import num_classes, device, dtype
from parameters import input_size


def init_weight_matrix():
    w = nn.Parameter(torch.empty((input_size, num_classes), device=device, dtype=dtype, requires_grad=True))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(w, -bound, bound)
    return w
