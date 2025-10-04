from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size, num_workers, shuffle):
    """
    Returns the dataloader.

    :param dataset: The dataset to use.
    :param batch_size: The batch size to use.
    :param num_workers: The number of workers to use.
    :param shuffle: Whether to shuffle the dataset.
    :return: The dataloader.
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    return dataloader
