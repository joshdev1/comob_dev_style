import torch

from parameters import BATCH_SIZE


def data_generator(ipds, spikes):
    perm = torch.randperm(spikes.shape[0])
    spikes = spikes[perm, :, :]
    ipds = ipds[perm]
    n, _, _ = spikes.shape
    n_batch = n//BATCH_SIZE
    for i in range(n_batch):
        x_local = spikes[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :]
        y_local = ipds[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        yield x_local, y_local
