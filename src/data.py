# data.py
from torchvision import datasets, transforms
import numpy as np
import torch

def get_mnist(root='../data', train=True, download=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST(root, train=train, download=download, transform=transform)
    return ds

def inject_label_noise(dataset, noise_rate, num_classes=10, seed=0):
    import random
    random.seed(seed)
    targets = list(dataset.targets)  # can be tensor or list
    n = len(targets)
    num_noisy = int(noise_rate * n)
    if num_noisy == 0:
        return dataset
    idx = np.random.choice(n, num_noisy, replace=False)
    for i in idx:
        orig = int(targets[i])
        choices = list(range(num_classes))
        choices.remove(orig)
        targets[i] = random.choice(choices)
    # keep same type (tensor/list)
    if isinstance(dataset.targets, torch.Tensor):
        dataset.targets = torch.tensor(targets, dtype=dataset.targets.dtype)
    else:
        dataset.targets = targets
    return dataset
