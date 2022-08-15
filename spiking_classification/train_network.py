import numpy as np
import torch
from torch import nn

from generating_stimuli.random_ipd_input import RandomIpdInput
from graph import plot_loss_function_over_time
from non_spiking_classification.analyse_results import analyse
from non_spiking_classification.data_generator import data_generator
from non_spiking_classification.weights import init_weight_matrix
from parameters import num_samples, lr, input_size, num_hidden, num_classes, nb_epochs, batch_size, n_testing_batches
from spiking_classification.snn import snn

training_data = RandomIpdInput(num_samples)
ipds, spikes = training_data.generate()

W1 = init_weight_matrix(input_size, num_hidden)
W2 = init_weight_matrix(num_hidden, num_classes)

optimizer = torch.optim.Adam([W1, W2], lr=lr)
log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()

print(f"Want loss for epoch 1 to be about {-np.log(1/num_classes):.2f}, multiply m by constant to get this")

loss_hist = []
for e in range(nb_epochs):
    local_loss = []
    for x_local, y_local in data_generator(training_data.discretise(ipds), spikes):
        output = snn(x_local, W1, W2) # Run network
        m = torch.mean(output, 1)  # Mean across time dimension
        loss = loss_fn(log_softmax_fn(m), y_local)
        local_loss.append(loss.item())
        # Update gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_hist.append(np.mean(local_loss))
    print("Epoch %i: loss=%.5f" % (e+1, np.mean(local_loss)))

plot_loss_function_over_time(loss_hist)

print(f"Chance accuracy level: {100*1/num_classes:.1f}%")
run_func = lambda x: snn(x, W1, W2)
analyse(ipds, spikes, 'Train', run=run_func)
test_data = RandomIpdInput(batch_size*n_testing_batches)
ipds_test, spikes_test = test_data.generate()
analyse(ipds_test, spikes_test, 'Test', run=run_func)
