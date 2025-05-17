import torch
import torchaudio

def prepare_audio(audio: torch.Tensor, sample_rate: int, start: int, end: int, noise_ratio, target_sr=16000, fixed_length=160000) -> torch.Tensor:
    """
    Grabs the input waveform, converts it to mono, resamples it to 16kHz takes only the given interval of time and applies Gaussian Noise based on the ratio.

    :param audio: The waveform of the audio, needs to be a torch.tensor
    :param sample_rate: The sample rate of the audio
    :param start: The start time of the audio, measured in 16kHz samples
    :param end: The end time of the audio, measured in 16kHz samples
    :param noise_ratio: The Gaussian noise ratio to be applied to the audio
    :param target_sr: The target sampling rate, default to 16kHz
    :param fixed_length: The fixed length of the audio, default to 160.000 (10 seconds of audio)

    :return: The waveform of the audio.
    """
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


