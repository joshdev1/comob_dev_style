import numpy as np
import torch
import torch.nn as nn

from graph import plot_loss_function_over_time
from generating_stimuli.random_ipd_input import RandomIpdInput
from non_spiking_classification.analyse_results import analyse
from non_spiking_classification.data_generator import data_generator
from non_spiking_classification.membrane import get_membrane_potential
from non_spiking_classification.weights import init_weight_matrix
from parameters import num_classes, num_samples, nb_epochs, lr, batch_size, n_testing_batches

training_data = RandomIpdInput(num_samples)
ipds, spikes = training_data.generate()


weights = init_weight_matrix()
optimizer = torch.optim.Adam([weights], lr=lr)
log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()

print(f"Want loss for epoch 1 to be about {-np.log(1/num_classes):.2f}, multiply m by constant to get this")

loss_hist = []
for e in range(nb_epochs):
    local_loss = []
    for x_local, y_local in data_generator(training_data.discretise(ipds), spikes):
        output = get_membrane_potential(x_local, weights)
        m = torch.sum(output, 1)*0.01
        loss = loss_fn(log_softmax_fn(m), y_local)
        local_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_hist.append(np.mean(local_loss))
    print("Epoch %i: loss=%.5f" % (e+1, np.mean(local_loss)))

plot_loss_function_over_time(loss_hist)

# running analysis function
print(f"Chance accuracy level: {100*1/num_classes:.1f}%")
run_func = lambda x: get_membrane_potential(x, weights)
analyse(ipds, spikes, 'Train', run=run_func)
test_data = RandomIpdInput(batch_size*n_testing_batches)
ipds_test, spikes_test = test_data.generate()
analyse(ipds_test, spikes_test, 'Test', run=run_func)

