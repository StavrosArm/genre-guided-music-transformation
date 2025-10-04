from torch.utils.data import Dataset
import torchaudio
from utils.audio_utils import prepare_audio
import warnings
import pandas as pd
import os


class FMADataset(Dataset):
    def __init__(self, config, mode="train", target_sr=16000):
        """
        csv_path: path to the metadata CSV
        audio_dir: base directory for audio files
        target_sr: target sample rate
        transform: optional additional transform (e.g. spectrogram)
        """
        self.config = config
        if mode == "train":
            csv_path = self.config.dataset.train.train_csv
        elif mode == "val":
            csv_path = self.config.dataset.val.val_csv
        elif mode == "test":
            csv_path = self.config.dataset.test.test_csv
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.df = pd.read_csv(csv_path)
        self.audio_dir = self.config.dataset.audio_dir
        self.target_sr = target_sr
        unique_labels = sorted(self.df["genre_top"].dropna().unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["path_to_audio"])
        try:
            waveform, sr = torchaudio.load(audio_path)

            audio = prepare_audio(
                waveform,
                sr,
                int(row["start"]),
                int(row["end"]),
                float(row["noise_ratio"]),
                target_sr=self.target_sr,
            )

            label = self.label_to_idx[row["genre_top"]]
            return audio, label
        except Exception as e:
            warnings.warn(f"Failed to load audio from {audio_path}: {e}")
            return self.__getitem__((idx + 1) % self.__len__())
