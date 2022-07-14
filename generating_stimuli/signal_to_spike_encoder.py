import numpy as np
from generating_stimuli.parameters import duration_steps, anf_per_ear, dt, envelope_power, rate_max, frequency


def signal_to_spike_encoder(ipd):
    num_samples = len(ipd)
    time = np.arange(duration_steps) * dt
    phi = 2 * np.pi * (frequency * time + np.random.rand())
    theta = np.zeros((num_samples, duration_steps, 2 * anf_per_ear))
    phase_delays = np.linspace(0, np.pi / 2, anf_per_ear)
    theta[:, :, :anf_per_ear] = phi[np.newaxis, :, np.newaxis] + phase_delays[np.newaxis, np.newaxis, :]
    theta[:, :, anf_per_ear:] = phi[np.newaxis, :, np.newaxis] + phase_delays[np.newaxis, np.newaxis, :] + ipd[:, np.newaxis, np.newaxis]
    spikes = np.random.rand(num_samples, duration_steps, 2 * anf_per_ear) < rate_max * dt * (
            0.5 * (1 + np.sin(theta))) ** envelope_power
    return spikes

