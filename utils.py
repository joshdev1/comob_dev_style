import torch


def check_if_gpu_is_available():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return device
    else:
        device = torch.device("cpu")
        return device
