import torch

from parameters import num_hidden, batch_size, device, dtype, duration_steps, num_classes, alpha
from spiking_classification.surrogate_gradient_descent import spike_fn


def run_network(input_spikes, w1, w2):
    s_rec = _input_to_hidden(input_spikes, w1)
    return _hidden_to_output(s_rec, w2)


def _input_to_hidden(input_spikes, w1):
    v = torch.zeros((batch_size, num_hidden), device=device, dtype=dtype)
    s = torch.zeros((batch_size, num_hidden), device=device, dtype=dtype)
    s_rec = [s]
    h = _update_weights_with_spikes(input_spikes, w1)
    for t in range(duration_steps - 1):
        new_v = (alpha * v + h[:, t, :]) * (1 - s)
        s = spike_fn(v - 1)
        v = new_v
        s_rec.append(s)
    return s_rec


def _hidden_to_output(s_rec, w2):
    v = torch.zeros((batch_size, num_classes), device=device, dtype=dtype)
    v_rec = [v]
    h = _update_weights_with_spikes(torch.stack(s_rec, dim=1), w2)
    for t in range(duration_steps - 1):
        v = alpha * v + h[:, t, :]
        v_rec.append(v)
    return torch.stack(v_rec, dim=1)


def _update_weights_with_spikes(spikes, weights):
    return torch.einsum("abc,cd->abd", (spikes, weights))
