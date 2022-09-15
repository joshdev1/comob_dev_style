import torch

from parameters import NUM_HIDDEN, BATCH_SIZE, DEVICE, DTYPE, DURATION_STEPS, NUM_CLASSES, alpha
from spiking_classification.surrogate_gradient_descent import spike_fn


def run_network(input_spikes, w1, w2):
    s_rec = _input_to_hidden(input_spikes, w1)
    return _hidden_to_output(s_rec, w2)


def _input_to_hidden(input_spikes, w1):
    v = torch.zeros((BATCH_SIZE, NUM_HIDDEN), device=DEVICE, dtype=DTYPE)
    s = torch.zeros((BATCH_SIZE, NUM_HIDDEN), device=DEVICE, dtype=DTYPE)
    s_rec = [s]
    h = _update_weights_with_spikes(input_spikes, w1)
    for t in range(DURATION_STEPS - 1):
        new_v = (alpha * v + h[:, t, :]) * (1 - s)
        s = spike_fn(v - 1)
        v = new_v
        s_rec.append(s)
    return s_rec


def _hidden_to_output(s_rec, w2):
    v = torch.zeros((BATCH_SIZE, NUM_CLASSES), device=DEVICE, dtype=DTYPE)
    v_rec = [v]
    h = _update_weights_with_spikes(torch.stack(s_rec, dim=1), w2)
    for t in range(DURATION_STEPS - 1):
        v = alpha * v + h[:, t, :]
        v_rec.append(v)
    return torch.stack(v_rec, dim=1)


def _update_weights_with_spikes(spikes, weights):
    return torch.einsum("abc,cd->abd", (spikes, weights))
