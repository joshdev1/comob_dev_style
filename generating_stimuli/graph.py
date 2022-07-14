import matplotlib.pyplot as plt
import numpy as np


def graph_stimuli(ipd, spikes):
    spikes = spikes.cpu()
    plt.figure(figsize=(10, 4), dpi=100)
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(spikes[i, :, :].T, aspect='auto', interpolation='nearest', cmap=plt.cm.gray_r)
        plt.title(f'True IPD = {int(ipd[i] * 180 / np.pi)} deg')
        if i >= 4:
            plt.xlabel('Time (steps)')
        if i % 4 == 0:
            plt.ylabel('Input neuron index')
    plt.tight_layout()
    plt.show()



