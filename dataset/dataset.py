from torch.utils.data import Dataset
import torchaudio
from utils.audio_utils import prepare_audio
import pandas as pd
import os


class FMADataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sr=16000):
        """
        csv_path: path to the metadata CSV
        audio_dir: base directory for audio files
        target_sr: target sample rate
        transform: optional additional transform (e.g. spectrogram)
        """
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        unique_labels = sorted(self.df['genre_top'].dropna().unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row['path_to_audio'])
        waveform, sr = torchaudio.load(audio_path)

        audio = prepare_audio(
            waveform,
            sr,
            int(row['start']),
            int(row['end']),
            float(row['noise_ratio']),
            target_sr=self.target_sr
        )

        label = self.label_to_idx[row['genre_top']]
        return audio, label
