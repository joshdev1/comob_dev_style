import numpy as np
import torch


def check_if_gpu_is_available():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device
    else:
        device = torch.device("cpu")
        return device


dtype = torch.float
device = check_if_gpu_is_available()
my_computer_is_slow = False

second = 1
ms = 1e-3
Hz = 1

# Simulation parameters
dt = 1 * ms  # large time step to make simulations run faster for tutorial
anf_per_ear = 100  # repeats of each ear with independent noise
envelope_power = 2  # higher values make sharper envelopes, easier
duration = .1 * second  # stimulus duration
duration_steps = int(np.round(duration / dt))
input_size = 2 * anf_per_ear
num_hidden = 30
beta = 5

# Stimulus parameters
frequency = 20 * Hz  # stimulus frequency
rate_max = 600 * Hz  # maximum Poisson firing rate


