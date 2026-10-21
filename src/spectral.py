"""Representaciones comparables; nunca normaliza cada señal por separado."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import librosa
import numpy as np

from .demucs_transcriber import AudioArtifact

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 128


@dataclass
class SpectralComparison:
    matrices: dict[str, np.ndarray]
    reference: float
    vmin: float
    vmax: float
    y_axis: str
    label: str


def spectral_comparison(artifacts: Mapping[str, AudioArtifact],
                        reference_audio: AudioArtifact,
                        representation: str = "log-Mel") -> SpectralComparison:
    """STFT en dB, Mel en potencia relativa o log-Mel en dB.

    Usa siempre el stem sin DSP como referencia, incluso cuando no está entre
    las señales visibles. Un stem silencioso usa referencia 1 para evitar NaN.
    """
    if representation not in ("STFT", "Mel", "log-Mel"):
        raise ValueError("Representación desconocida.")
    if not artifacts:
        raise ValueError("Se necesita al menos una señal para comparar.")
    original = reference_audio.mono(SAMPLE_RATE)

    def transform(artifact):
        mono = artifact.mono(SAMPLE_RATE)
        if mono.audio.shape != original.audio.shape:
            raise ValueError("Las señales deben corresponder al mismo fragmento temporal.")
        magnitude = np.abs(librosa.stft(mono.audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
        if representation == "STFT":
            return magnitude
        return librosa.feature.melspectrogram(
            S=magnitude ** 2, sr=SAMPLE_RATE, n_fft=N_FFT,
            n_mels=N_MELS, fmin=0, fmax=SAMPLE_RATE / 2)

    peak = float(transform(original).max())
    reference = peak if peak > 0 else 1.0
    matrices = {}
    for name, artifact in artifacts.items():
        native = transform(artifact)
        if representation == "STFT":
            matrices[name] = librosa.amplitude_to_db(native, ref=reference, amin=1e-12, top_db=None)
        elif representation == "log-Mel":
            matrices[name] = librosa.power_to_db(native, ref=reference, amin=1e-24, top_db=None)
        else:
            matrices[name] = native / reference
    is_db = representation != "Mel"
    upper = 0.0 if is_db else 1.0
    maximum = max(float(matrix.max()) for matrix in matrices.values())
    # La conversión float32 puede dejar el máximo de referencia apenas sobre 0 dB.
    if maximum > upper and not np.isclose(maximum, upper, atol=1e-5, rtol=1e-6):
        upper = maximum
    return SpectralComparison(
        matrices=matrices, reference=reference, vmin=-80.0 if is_db else 0.0,
        vmax=upper,
        y_axis="log" if representation == "STFT" else "mel",
        label=("dB respecto al máximo del stem sin DSP" if is_db
               else "Potencia Mel / máximo del stem sin DSP"),
    )
