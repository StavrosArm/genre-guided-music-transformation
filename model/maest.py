from types import SimpleNamespace
from typing import Tuple

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import numpy as np
import torch


class Maest(torch.nn.Module):
    def __init__(self, config):
        super(Maest, self).__init__()
        self.config = config
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.config.distortion.model_name,
                                                                      trust_remote_code=True)
        self.encoder = AutoModelForAudioClassification.from_pretrained(self.config.distortion.model_name,
                                                                       trust_remote_code=True).audio_spectrogram_transformer

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: tensor with the waveform

        :return: Tensor with the classification embedding
        """
        inputs = self.feature_extractor(audio, sampling_rate=self.config.distortion.sampling_rate, return_tensors="pt")

        with torch.no_grad():
            outputs = self.encoder(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def get_embeddings_for_fad_distance(self, config) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Implement reading logic from the directories, return both embeddings context.
        pass


if __name__ == "__main__":
    config = SimpleNamespace(
        distortion=SimpleNamespace(
            model_name="mtg-upf/discogs-maest-10s-pw-129e",
            sampling_rate=16000
        )
    )

    model = Maest(config)
    print(model)

    dummy_audio = torch.randn(int(config.distortion.sampling_rate))
    embedding = model.forward(dummy_audio)
    print(embedding)
