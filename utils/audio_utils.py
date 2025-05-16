import torch
import torchaudio


def prepare_audio(audio, sample_rate, start, end, noise_ratio, target_sr=16000):
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        audio = resampler(audio)

    audio = audio[:, start:end]

    if noise_ratio > 0:
        noise = torch.randn_like(audio) * noise_ratio
        audio = audio + noise

    return audio
