import numpy as np
import torch
from torch import nn

from non_spiking_classification.data_generator import data_generator
from parameters import num_classes, nb_epochs
from spiking_classification.snn import run_network


print(f"Want loss for epoch 1 to be about {-np.log(1/num_classes):.2f}, multiply m by constant to get this")

log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()


def train_network(w1, w2, training_data, ipds, spikes, optimizer):
    loss_hist = []
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in data_generator(training_data.discretise(ipds), spikes):
            output = run_network(x_local, w1, w2)
            loss = _cross_entropy(output, y_local, local_loss)
            _update_gradients(optimizer, loss)
        loss_hist.append(np.mean(local_loss))
        _display_progress(e, local_loss)
    return w1, w2, loss_hist


def _cross_entropy(output, y_local, local_loss):
    m = torch.mean(output, 1)  # Mean across time dimension
    loss = loss_fn(log_softmax_fn(m), y_local)
    local_loss.append(loss.item())
    return loss


def _update_gradients(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _display_progress(e, local_loss):
    print("Epoch %i: loss=%.5f" % (e + 1, np.mean(local_loss)))
