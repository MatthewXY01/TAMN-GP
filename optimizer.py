import torch

def init_optimizer(optim, params, lr, weight_decay):
    if optim=='Adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim=='SGD':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        raise KeyError("Unsupported optimizer: {}".format(optim))