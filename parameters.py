import numpy as np
import torch


def check_if_gpu_is_available():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        return DEVICE
    else:
        DEVICE = torch.device("cpu")
        return DEVICE


DTYPE = torch.float
DEVICE = check_if_gpu_is_available()
my_computer_is_slow = False

if my_computer_is_slow:
    BATCH_SIZE = 64
    n_training_batches = 64
else:
    BATCH_SIZE = 128
    n_training_batches = 128
N_TESTING_BATCHES = 32
NUM_SAMPLES = BATCH_SIZE * n_training_batches

second = 1
ms = 1e-3
Hz = 1

# Simulation parameters
dt = 1 * ms  # large time step to make simulations run faster for tutorial
ANF_PER_EAR = 100  # repeats of each ear with independent noise
ENVELOPE_POWER = 2  # higher values make sharper envelopes, easier
DURATION = .1 * second  # stimulus duration
DURATION_STEPS = int(np.round(DURATION / dt))
INPUT_SIZE = 2 * ANF_PER_EAR
NUM_HIDDEN = 30
tau = 20*ms
beta = 5
alpha = np.exp(-dt/tau)
NUM_CLASSES = 180//15  # num_classes = 12

# Training parameters
EPOCHS = 10  # quick, it won't have converged
LEARNING_RATE = 0.01  # learning rate

# Stimulus parameters
FREQUENCY = 20 * Hz  # stimulus frequency
RATE_MAX = 600 * Hz  # maximum Poisson firing rate


