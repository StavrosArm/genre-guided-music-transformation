import torch
import numpy as np
import random


def set_seed(config):
    """
    Sets all the random seeds to the one provided in the configuration file.

    :param config: The configuration object
    """
    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
