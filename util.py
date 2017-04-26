import torch
import torch.nn as nn

def get_num_parameters(model):
    N = 0
    for p in model.parameters():
        N += p.numel()

    return N
