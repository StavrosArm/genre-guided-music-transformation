import torch
import torchaudio

def prepare_audio(audio, sample_rate, start, end, noise_ratio, target_sr=16000, fixed_length=160000):
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        audio = resampler(audio)

    audio = audio[:, start:end]

    if audio.shape[1] < fixed_length:
        pad_size = fixed_length - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, pad_size))
    elif audio.shape[1] > fixed_length:
        audio = audio[:, :fixed_length]

    if noise_ratio > 0:
        noise = torch.randn_like(audio) * noise_ratio
        audio = audio + noise

    return audio.squeeze(0)

