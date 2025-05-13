import torch.nn as nn

def get_loss_function(config):
    """
    Returns the loss function

    :param config: The configuration object containing the loss type
    :return: The loss function, bce or cross entropy based on the configuration.
    """
    if config.loss.type == "cross-entropy":
        return nn.CrossEntropyLoss()
    elif config.loss.type == "bce":
        return nn.BCEWithLogitsLoss()