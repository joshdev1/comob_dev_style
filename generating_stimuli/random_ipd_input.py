import numpy as np
import torch

from parameters import DTYPE, DEVICE, DURATION_STEPS, FREQUENCY, dt, ANF_PER_EAR, RATE_MAX, ENVELOPE_POWER, NUM_CLASSES


class RandomIpdInput:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.ipd = np.random.rand(self.num_samples) * np.pi - np.pi / 2
        self.time = np.arange(DURATION_STEPS) * dt
        self.phi = 2 * np.pi * (FREQUENCY * self.time + np.random.rand())
        self.theta = np.zeros((self.num_samples, DURATION_STEPS, 2 * ANF_PER_EAR))
        self.phase_delays = np.linspace(0, np.pi / 2, ANF_PER_EAR)

    def generate(self):
        self._add_phase_delay()
        return self._convert_to_tensor(self.ipd), self._convert_to_tensor(self._poisson_spike_train())

    def _add_phase_delay(self):
        self.theta[:, :, :ANF_PER_EAR] = self.phi[np.newaxis, :, np.newaxis] + self.phase_delays[np.newaxis, np.newaxis,
                                                                                                 :]
        self.theta[:, :, ANF_PER_EAR:] = self.phi[np.newaxis, :, np.newaxis] + self.phase_delays[np.newaxis, np.newaxis,
                                                                                                 :] + self.ipd[:,
                                                                                                      np.newaxis,
                                                                                                      np.newaxis]

    def _poisson_spike_train(self):
        return np.random.rand(self.num_samples, DURATION_STEPS, 2 * ANF_PER_EAR) < RATE_MAX * dt * (
                0.5 * (1 + np.sin(self.theta))) ** ENVELOPE_POWER

    @staticmethod
    def discretise(ipds):
        return ((ipds + np.pi / 2) * NUM_CLASSES / np.pi).long()

    @staticmethod
    def continuise(ipd_indices):
        return (ipd_indices + 0.5) / NUM_CLASSES * np.pi - np.pi / 2

    @staticmethod
    def _convert_to_tensor(input_array):
        return torch.tensor(input_array, device=DEVICE, dtype=DTYPE)
