import numpy as np
from torch.onnx._internal import onnxruntime

from utils.audio_utils import prepare_audio
import torchaudio

GENRE_LABELS = {
    0: "electronic",
    1: "experimental",
    2: "hip-hop",
    3: "instrumental",
    4: "international",
    5: "jazz",
    6: "pop",
    7: "rock"
}

def predict_song(audio_path: str):
    """
    Predicts the genre of the song using onnx for cpu acceleration.

    :param audio_path: The csv file containing the audio data
    :return: A string with the predicted genre
    """
    audio, sr = torchaudio.load(audio_path)
    audio = prepare_audio(audio, sr,0, 160000, 0.0).numpy().astype(np.float32)

    input = audio[np.newaxis, np.newaxis, :]

    ort_session = onnxruntime.InferenceSession()

    input_name = ort_session.get_inputs()[0].name
    output = ort_session.run(None, {input_name: input})

    genre = np.argmax(output[0])
    return f'The genre of the song is {GENRE_LABELS[genre]}', genre











