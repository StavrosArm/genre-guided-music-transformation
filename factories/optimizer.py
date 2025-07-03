from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR


def get_optimizer(config, model_params):
    """
    Creates and returns an optimizer based on the configuration.

    :param config: Configuration object containing optimizer settings (e.g., name, lr, weight_decay).
    :param model_params: Parameters of the model to optimize (e.g., model.parameters()).
    :return: A torch.optim optimizer instance (e.g., AdamW).
    :raises ValueError: If the optimizer name is unsupported.
    """
    optimizer_name = config.optimizer.optimizer_name.lower()
    lr = float(config.optimizer.lr)
    weight_decay = float(config.optimizer.weight_decay or 0)

    if optimizer_name == "adamw":
        return AdamW(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(config, steps_per_epoch, optimizer):
    """
        Creates and returns a linear learning rate scheduler with optional warmup using PyTorch's LambdaLR.

        :param config: Configuration object containing scheduler settings, such as scheduler type,
                       number of epochs, warmup steps, and steps per epoch.
        :param optimizer: The optimizer instance to apply the learning rate schedule to.
        :param steps_per_epoch: The number of steps per epoch.
        :return: A torch.optim.lr_scheduler.LambdaLR scheduler instance, or None if scheduler is disabled.
        :raises ValueError: If the scheduler name is unsupported.
    """
    scheduler_name = config.optimizer.scheduler.lower()
    total_epochs = config.training.epochs

    if scheduler_name == "LambdaLR":
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = config.optimizer.warmup_steps * total_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif scheduler_name == "StepLR":
        step_size = config.optimizer.cut_epochs * steps_per_epoch
        return StepLR(optimizer, step_size=step_size, gamma = float(config.optimizer.gamma))

    elif scheduler_name in ["", "none", None]:
        return None

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
