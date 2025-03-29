import librosa
import numpy as np
import torch
from typing import List

class MelExtractor:
    def __init__(self, config):
        self.config = config
        self.n_fft = config.feature_extractor.n_fft
        self.max_length = config.feature_extractor.max_length
        self.sample_rate = config.feature_extractor.sample_rate
        self.n_mels = config.feature_extractor.n_mels

    def _extract_mel(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Extract mel spectrogram from a single audio signal.
        Args:
            audio (np.ndarray): shape (samples,)
            sr (int): sample rate of audio

        Returns:
            torch.Tensor: shape (n_mels, max_length)
        """
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.n_fft // 4,
            n_mels=self.n_mels,
            power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Pad or trim to max_length
        if mel_db.shape[1] < self.max_length:
            pad_width = self.max_length - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :self.max_length]

        return torch.tensor(mel_db, dtype=torch.float32)

    def __call__(self, audio_paths: List[str]) -> torch.Tensor:
        """
        Load audio files and return batched mel spectrograms.

        Args:
            audio_paths (List[str]): paths to wav files

        Returns:
            torch.Tensor: shape (batch_size, n_mels, max_length)
        """
        mel_batch = []
        for path in audio_paths:
            y, sr = librosa.load(path, sr=None)
            mel = self._extract_mel(y, sr)
            mel_batch.append(mel)

        return torch.stack(mel_batch)
