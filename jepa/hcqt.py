# Copyright 2026 AcusPy contributors
#
# HCQT input representation faithful to Bittner et al. (2022),
# "A Lightweight Instrument-Agnostic Model for Polyphonic Note
# Transcription and Multipitch Estimation" (basic-pitch NMP).
#
# Run interactively with `marimo edit jepa/hcqt.py` or import the
# definitions from any other notebook: `import hcqt`.

import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    import sys
    from pathlib import Path

    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).parent))

    # Constants from basic-pitch (Spotify).
    SR = 22_050
    FFT_HOP = 512
    BASE_FMIN = 32.70319566257483  # C1
    ANNOTATION_N_SEMITONES = 88
    BINS_PER_SEMITONE = 3
    N_HARMONICS = 8
    HARMONICS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    N_FREQ_BINS_CONTOURS = 256
    CQT_N_SEMITONES = min(ANNOTATION_N_SEMITONES + 36, 99)  # Nyquist limit
    CQT_N_BINS = CQT_N_SEMITONES * BINS_PER_SEMITONE  # 297
    FRAMES_PER_SECOND = SR / FFT_HOP  # ~43.07


@app.function
def harmonic_shifts(bins_per_semitone=3, harmonics=(0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)):
    """Per-harmonic frequency shifts in CQT bins, exactly as basic-pitch's
    HarmonicStacking: round(12 * bins_per_semitone * log2(h))."""
    return tuple(
        int(round(12 * bins_per_semitone * np.log2(float(h)))) for h in harmonics
    )


@app.function
def normalized_log(magnitude):
    """basic-pitch's NormalizedLog: magnitude -> dB scaled 0-1 per sample.

    magnitude: [..., F, T] tensor or array. Returns same shape in [0, 1].
    """
    if isinstance(magnitude, np.ndarray):
        magnitude = torch.from_numpy(magnitude)
    else:
        magnitude = torch.as_tensor(magnitude)
    power = magnitude.square()
    log_power = 10.0 * torch.log10(power + 1e-10)
    dims = (-2, -1)
    lo = log_power.amin(dim=dims, keepdim=True)
    hi = (log_power - lo).amax(dim=dims, keepdim=True)
    return (log_power - lo).div(hi.clamp_min(1e-10))


@app.function
def harmonic_stacking(cqt, n_output_freqs=256):
    """Stack harmonically-shifted copies of the CQT along the channel axis.

    cqt: [..., n_bins, T] log-CQT with n_bins = CQT_N_BINS.
    Returns: [..., len(HARMONICS), n_output_freqs, T].
    Channel h holds cqt[f + shift_h] (zeros where out of range).
    """
    shifts = harmonic_shifts()
    freqs = torch.arange(n_output_freqs, device=cqt.device)
    channels = []
    for shift in shifts:
        idx = freqs + shift
        valid = (idx >= 0) & (idx < cqt.shape[-2])
        safe = idx.clamp(0, cqt.shape[-2] - 1)
        channels.append(cqt[..., safe, :] * valid[..., None])
    return torch.stack(channels, dim=-3)


@app.function
def cqt_magnitude(audio, sr=22_050, n_bins=297):
    """Magnitude CQT of a 1-D waveform, matching basic-pitch parameters."""
    import librosa

    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    cqt = librosa.cqt(
        audio.astype(np.float32),
        sr=SR,
        hop_length=FFT_HOP,
        fmin=BASE_FMIN,
        n_bins=n_bins,
        bins_per_octave=12 * BINS_PER_SEMITONE,
    )
    return np.abs(cqt).astype(np.float32)  # [n_bins, T]


@app.function
def hcqt_from_audio(audio, sr=22_050):
    """Full HCQT of a waveform: returns [8, 256, T] in [0, 1]."""
    return harmonic_stacking(
        normalized_log(torch.from_numpy(cqt_magnitude(audio, sr)))
    ).numpy()


@app.function
def audio_to_target_frames(n_samples):
    """Number of CQT frames produced by n_samples of audio."""
    return 1 + int(np.ceil(n_samples / FFT_HOP))


@app.cell
def _():
    """Doc cell: sanity check the stacking shifts."""
    import marimo as mo

    mo.md(
        rf"""
        # HCQT (basic-pitch)

        - `{SR} Hz`, hop `{FFT_HOP}` ({FRAMES_PER_SECOND:.2f} frames/s ≈ 11.6 ms)
        - CQT: `{CQT_N_BINS}` bins, `{BINS_PER_SEMITONE}` bins/semitone, fmin = C1
        - Harmonic stacking shifts (bins): `{harmonic_shifts()}`
        - Output HCQT: `{N_HARMONICS} x {N_FREQ_BINS_CONTOURS} x T`
        """
    )
    return (mo,)


@app.cell
def _():
    """Interactive sanity check of the HCQT pipeline."""
    import numpy as _np

    _t = _np.arange(2 * SR) / SR
    _audio = 0.5 * _np.sin(2 * _np.pi * 220.0 * _t)  # A3
    _h = hcqt_from_audio(_audio)
    assert _h.shape == (8, N_FREQ_BINS_CONTOURS, 87), _h.shape
    assert 0.0 <= float(_h.min()) and float(_h.max()) <= 1.0
    print("HCQT shape:", _h.shape, "range:", float(_h.min()), float(_h.max()))
    return


if __name__ == "__main__":
    app.run()
