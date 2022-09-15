import torch

from generating_stimuli.random_ipd_input import RandomIpdInput
from graph import plot_loss_function_over_time
from non_spiking_classification.analyse_results import analyse
from non_spiking_classification.weights import init_weight_matrix
from parameters import NUM_SAMPLES, INPUT_SIZE, NUM_HIDDEN, NUM_CLASSES, LEARNING_RATE, BATCH_SIZE, N_TESTING_BATCHES
from spiking_classification.snn import run_network
from spiking_classification.train_network import train_network

training_data = RandomIpdInput(NUM_SAMPLES)
ipds, spikes = training_data.generate()

w1 = init_weight_matrix(INPUT_SIZE, NUM_HIDDEN)
w2 = init_weight_matrix(NUM_HIDDEN, NUM_CLASSES)

optimizer = torch.optim.Adam([w1, w2], lr=LEARNING_RATE)

optimised_w1, optimised_w2, loss_hist = train_network(w1, w2, training_data, ipds, spikes, optimizer)

plot_loss_function_over_time(loss_hist)

print(f"Chance accuracy level: {100*1/NUM_CLASSES:.1f}%")
run_func = lambda x: run_network(x, optimised_w1, optimised_w2)
analyse(ipds, spikes, 'Train', run=run_func)
test_data = RandomIpdInput(BATCH_SIZE*N_TESTING_BATCHES)
ipds_test, spikes_test = test_data.generate()
analyse(ipds_test, spikes_test, 'Test', run=run_func)
