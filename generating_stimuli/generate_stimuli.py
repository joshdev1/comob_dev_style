import numpy as np
import torch

from generating_stimuli.parameters import device, dtype
from generating_stimuli.signal_to_spike_encoder import signal_to_spike_encoder


def random_ipd_input_signal(num_samples, tensor=True):
    ipd = np.random.rand(num_samples) * np.pi - np.pi / 2
    spikes = signal_to_spike_encoder(ipd)
    if tensor:
        ipd = torch.tensor(ipd, device=device, dtype=dtype)
        spikes = torch.tensor(spikes, device=device, dtype=dtype)
    return ipd, spikes
