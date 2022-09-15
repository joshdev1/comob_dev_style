import torch

from parameters import BATCH_SIZE, NUM_CLASSES, DEVICE, DTYPE, DURATION_STEPS, alpha


def get_membrane_potential(input_spikes, weights):
    v, v_rec = _set_initial_membrane_voltage()
    h = _get_input(input_spikes, weights)
    return _calculate_voltages(v, v_rec, h)


def _set_initial_membrane_voltage():
    v = torch.zeros((BATCH_SIZE, NUM_CLASSES), device=DEVICE, dtype=DTYPE)
    v_rec = [v]
    return v, v_rec


def _get_input(input_spikes, weights):
    return torch.einsum("abc,cd->abd", (input_spikes, weights))


def _calculate_voltages(v, v_rec, h):
    for t in range(DURATION_STEPS - 1):
        v = alpha*v + h[:, t, :]
        v_rec.append(v)
    return torch.stack(v_rec, dim=1)  # (batch_size, duration_steps, num_classes)


# import torch
#
# from parameters import batch_size, num_classes, device, dtype, duration_steps, alpha
#
#
# class Membrane:
#     def __init__(self,):
#         self.v = torch.zeros((batch_size, num_classes), device=device, dtype=dtype)
#         self.v_rec = [self.v]
#
#     def get_membrane_potential(self, input_spikes, weights):
#         h = self._get_input(input_spikes, weights)
#         return self._calculate_voltages(h)
#
#     @staticmethod
#     def _get_input(input_spikes, weights):
#         return torch.einsum("abc,cd->abd", (input_spikes, weights))
#
#     def _calculate_voltages(self, h):
#         for t in range(duration_steps - 1):
#             v = alpha * self.v + h[:, t, :]
#             self.v_rec.append(v)
#         return torch.stack(self.v_rec, dim=1)  # (batch_size, duration_steps, num_classes)
#