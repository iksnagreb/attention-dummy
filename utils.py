# Python's builtin random number generators
import random
# Probably the most important one: The PyTorch random number generator
import torch
# Numpy has a default random number generator as well
import numpy as np


# Seeds all relevant random number generators to the same seed for
# reproducibility
def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
