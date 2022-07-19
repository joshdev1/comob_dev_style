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

if my_computer_is_slow:
    batch_size = 64
    n_training_batches = 64
else:
    batch_size = 128
    n_training_batches = 128
n_testing_batches = 32
num_samples = batch_size * n_training_batches

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
tau = 20*ms
beta = 5
alpha = np.exp(-dt/tau)
num_classes = 180//15  # num_classes = 12

# Training parameters
nb_epochs = 10  # quick, it won't have converged
lr = 0.01  # learning rate

# Stimulus parameters
frequency = 20 * Hz  # stimulus frequency
rate_max = 600 * Hz  # maximum Poisson firing rate


