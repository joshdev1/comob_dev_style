import numpy as np
import torch
import torch.nn as nn

from graph import plot_loss_function_over_time
from generating_stimuli.random_ipd_input import RandomIpdInput
from non_spiking_classification.membrane import membrane_only
from non_spiking_classification.weights import init_weight_matrix
from parameters import num_classes, batch_size, num_samples, nb_epochs, lr


def data_generator(ipds, spikes):
    perm = torch.randperm(spikes.shape[0])
    spikes = spikes[perm, :, :]
    ipds = ipds[perm]
    n, _, _ = spikes.shape
    n_batch = n//batch_size
    for i in range(n_batch):
        x_local = spikes[i*batch_size:(i+1)*batch_size, :, :]
        y_local = ipds[i*batch_size:(i+1)*batch_size]
        yield x_local, y_local


training_data = RandomIpdInput(num_samples)
ipds, spikes = training_data.generate()


weights = init_weight_matrix()

# Optimiser and loss function
optimizer = torch.optim.Adam([weights], lr=lr)
log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()  # negative log likelihood loss

print(f"Want loss for epoch 1 to be about {-np.log(1/num_classes):.2f}, multiply m by constant to get this")

loss_hist = []
for e in range(nb_epochs):
    local_loss = []
    for x_local, y_local in data_generator(training_data.discretise(ipds), spikes):
        # Run network
        output = membrane_only(x_local, weights)
        # Compute cross entropy loss
        m = torch.sum(output, 1)*0.01  # Sum time dimension
        loss = loss_fn(log_softmax_fn(m), y_local)
        local_loss.append(loss.item())
        # Update gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_hist.append(np.mean(local_loss))
    print("Epoch %i: loss=%.5f" % (e+1, np.mean(local_loss)))

plot_loss_function_over_time(loss_hist)






