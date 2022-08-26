import torch

from generating_stimuli.random_ipd_input import RandomIpdInput
from graph import plot_loss_function_over_time
from non_spiking_classification.analyse_results import analyse
from non_spiking_classification.weights import init_weight_matrix
from parameters import num_samples, input_size, num_hidden, num_classes, lr, batch_size, n_testing_batches
from spiking_classification.snn import run_network
from spiking_classification.train_network import train_network

training_data = RandomIpdInput(num_samples)
ipds, spikes = training_data.generate()

w1 = init_weight_matrix(input_size, num_hidden)
w2 = init_weight_matrix(num_hidden, num_classes)

optimizer = torch.optim.Adam([w1, w2], lr=lr)

optimised_w1, optimised_w2, loss_hist = train_network(w1, w2, training_data, ipds, spikes, optimizer)

plot_loss_function_over_time(loss_hist)

print(f"Chance accuracy level: {100*1/num_classes:.1f}%")
run_func = lambda x: run_network(x, optimised_w1, optimised_w2)
analyse(ipds, spikes, 'Train', run=run_func)
test_data = RandomIpdInput(batch_size*n_testing_batches)
ipds_test, spikes_test = test_data.generate()
analyse(ipds_test, spikes_test, 'Test', run=run_func)
