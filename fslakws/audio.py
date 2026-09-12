"""Audio loading and preprocessing for FSLAKWS."""

import soundfile as sf
import torch
import torchaudio

TARGET_SR = 16000
WINDOW_SECONDS = 1.0


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load a wav file. Returns (waveform, sample_rate)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channels, samples)
    return waveform, sr


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Collapse to a single channel."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def resample(waveform: torch.Tensor, sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Resample to target_sr if needed."""
    if sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    return resampler(waveform)


def normalize(waveform: torch.Tensor) -> torch.Tensor:
    """Peak-normalize to [-1, 1]."""
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def preprocess(path: str) -> torch.Tensor:
    """Load -> mono -> resample -> normalize. Returns 1D tensor at 16kHz."""
    waveform, sr = load_audio(path)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sr)
    waveform = normalize(waveform)
    return waveform.squeeze(0)


def make_windows(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SR,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = 0.25,
) -> list[dict]:
    """Slice audio into overlapping fixed-length windows."""
    win_len = int(window_seconds * sample_rate)
    hop_len = int(hop_seconds * sample_rate)
    total_len = waveform.shape[0]

    windows = []
    start_sample = 0
    while start_sample < total_len:
        end_sample = start_sample + win_len
        chunk = waveform[start_sample:end_sample]

        if chunk.shape[0] < win_len:
            pad_amount = win_len - chunk.shape[0]
            chunk = torch.nn.functional.pad(chunk, (0, pad_amount))

        windows.append({
            "audio": chunk,
            "start": start_sample / sample_rate,
            "end": (start_sample + win_len) / sample_rate,
        })

        if end_sample >= total_len:
            break
        start_sample += hop_len

    return windows