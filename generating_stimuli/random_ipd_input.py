import numpy as np
import torch

from parameters import dtype, device, duration_steps, frequency, dt, anf_per_ear, rate_max, envelope_power, num_classes


class RandomIpdInput:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.ipd = np.random.rand(self.num_samples) * np.pi - np.pi / 2
        self.time = np.arange(duration_steps) * dt
        self.phi = 2 * np.pi * (frequency * self.time + np.random.rand())
        self.theta = np.zeros((self.num_samples, duration_steps, 2 * anf_per_ear))
        self.phase_delays = np.linspace(0, np.pi / 2, anf_per_ear)

    def generate(self):
        self._add_phase_delay()
        return self._convert_to_tensor(self.ipd), self._convert_to_tensor(self._poisson_spike_train())

    def _add_phase_delay(self):
        self.theta[:, :, :anf_per_ear] = self.phi[np.newaxis, :, np.newaxis] + self.phase_delays[np.newaxis, np.newaxis,
                                                                               :]
        self.theta[:, :, anf_per_ear:] = self.phi[np.newaxis, :, np.newaxis] + self.phase_delays[np.newaxis, np.newaxis,
                                                                               :] + self.ipd[:,
                                                                                    np.newaxis,
                                                                                    np.newaxis]

    def _poisson_spike_train(self):
        return np.random.rand(self.num_samples, duration_steps, 2 * anf_per_ear) < rate_max * dt * (
                0.5 * (1 + np.sin(self.theta))) ** envelope_power

    @staticmethod
    def discretise(ipds):
        return ((ipds + np.pi / 2) * num_classes / np.pi).long()

    @staticmethod
    def continuise(ipd_indices):
        return (ipd_indices + 0.5) / num_classes * np.pi - np.pi / 2

    @staticmethod
    def _convert_to_tensor(input_array):
        return torch.tensor(input_array, device=device, dtype=dtype)
