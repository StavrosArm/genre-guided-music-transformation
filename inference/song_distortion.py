import os
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from model.model import MusicModel

from factories.optimizer import get_optimizer

from utils.audio_utils import prepare_audio
from utils.seed import set_seed

GENRE_LABELS = {
    "electronic": 0,
    "experimental": 1,
    "hip-hop": 2,
    "instrumental": 3,
    "international": 4,
    "jazz": 5,
    "pop": 6,
    "rock": 7
}


def distort_song(config):
    """
    Takes waveforms and distorts the original signal to match a specific genre. Uses the KL-Divergence between the target class and the
    model's output. Additionally, It uses norm 2 between the original signal and the distorted one to avoid "bad" outputs.

    :param config: The configuration object
    :return: None
    """
    set_seed(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MusicModel(config).to(device)
    model.load_state_dict(torch.load(f"{config.training.checkpoint_dir}/best_model.pt", map_location=device))
    model.eval()

    df = pd.read_csv(config.distortion.audios_csv)
    audio_dir = config.distortion.audio_dir
    target_dir = config.distortion.target_dir

    os.makedirs(target_dir, exist_ok=True)

    target_genre = config.distortion.genre
    target_index = GENRE_LABELS[target_genre]
    target_onehot = F.one_hot(torch.tensor([target_index]), num_classes=len(GENRE_LABELS)).float().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    norm_lambda = float(config.distortion.norm_lambda)
    threshold = float(config.distortion.threshold)
    max_steps = int(config.distortion.max_number_of_steps)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Distorting songs"):
        filename = row["path"]
        path = os.path.join(audio_dir, filename)

        waveform, sr = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        waveform = prepare_audio(waveform, sr, 0, 160000, 0.0)
        waveform = waveform.to(device)
        original_waveform = waveform.clone().detach()
        waveform = waveform.clone().detach().requires_grad_(True)

        optimizer = get_optimizer(config.distortion, [waveform])

        for step in range(max_steps):
            optimizer.zero_grad()
            outputs = model(waveform)

            ce_loss = criterion(outputs, target_onehot)
            proximity_loss = norm_lambda * F.mse_loss(waveform, original_waveform)
            total_loss = ce_loss + proximity_loss

            probs = F.softmax(outputs, dim=1)
            target_prob = probs[0, target_index].item()

            if target_prob >= threshold:
                break

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                waveform.clamp(-1.0, 1.0)

            print(
                f"[{step}] CE: {ce_loss.item():.4f}, Proximity: {proximity_loss.item():.4f}, Total: {total_loss.item():.4f}")
            print(f"Waveform diff: {F.mse_loss(waveform, original_waveform).item():.6f}")

        save_path = os.path.join(target_dir, os.path.basename(filename))
        torchaudio.save(save_path, waveform.detach().cpu(), 16000)

    print(f"Distorted waveforms saved to {target_dir}")
