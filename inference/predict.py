from utils.mel_extractor import MelExtractor
from model.model import MusicModel
import torch
from config import Config
import numpy as np

def predict_song(path: str, config: Config = "./config.yaml" ) -> np.ndarray:
    """
    :param path: the song's path
    :param config: the configuration file, defaults to "./config.yaml"
    :returns:  the model's prediction in numpy array
    """
    mel_extractor = MelExtractor(config)
    mel = mel_extractor([path])
    model = MusicModel(config)
    prediction = model(*mel).numpy()
    return prediction
