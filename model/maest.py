from types import SimpleNamespace
from typing import Tuple
import torchaudio

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

    import torchaudio

    def get_embeddings_for_fad_distance(
            self, audio_ref_path: str, audio_gen_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads audio from file paths, processes them, and returns full hidden states
        without the batch dimension.

        :param audio_ref_path: Path to reference audio file
        :param audio_gen_path: Path to generated audio file
        :return: Tuple of numpy arrays (hidden_states_ref, hidden_states_gen) with shape (seq_len, hidden_dim)
        """

        target_sr = self.config.distortion.sampling_rate

        def load_and_resample(path: str) -> torch.Tensor:
            waveform, sr = torchaudio.load(path)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample if needed
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
            return waveform.squeeze()  # remove channel dimension

        audio_ref = load_and_resample(audio_ref_path)
        audio_gen = load_and_resample(audio_gen_path)

        # Feature extraction and encoding
        inputs_ref = self.feature_extractor(audio_ref, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            outputs_ref = self.encoder(**inputs_ref)
        hidden_states_ref = outputs_ref.last_hidden_state.squeeze(0).detach().cpu().numpy()

        inputs_gen = self.feature_extractor(audio_gen, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            outputs_gen = self.encoder(**inputs_gen)
        hidden_states_gen = outputs_gen.last_hidden_state.squeeze(0).detach().cpu().numpy()

        return hidden_states_ref, hidden_states_gen


if __name__ == "__main__":
    config = SimpleNamespace(
        distortion=SimpleNamespace(
            model_name="mtg-upf/discogs-maest-10s-pw-129e",
            sampling_rate=16000
        )
    )

    model = Maest(config)

    dummy_audio_ref = torch.randn(int(config.distortion.sampling_rate))
    dummy_audio_gen = torch.randn(int(config.distortion.sampling_rate))

    # hidden_ref, hidden_gen = model.get_embeddings_for_fad_distance(dummy_audio_ref, dummy_audio_gen)

    print("Reference hidden states shape:", hidden_ref.shape)
    print("Generated hidden states shape:", hidden_gen.shape)
