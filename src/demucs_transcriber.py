"""HT-Demucs → variantes DSP por stem → Basic Pitch.

Los arrays estéreo usan (canales, muestras); los mono, (muestras,).
La separación y las variantes se pueden reutilizar sin repetir los modelos.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import librosa
import numpy as np
import soundfile as sf

from .transcriber import AudioPreProcessor, BasicPitchTranscriber, NotePostProcessor


def available_devices() -> list[str]:
    """Dispositivos de Demucs disponibles en este proceso, sin cargar modelos."""
    import torch

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _identifier(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Los nombres deben usar letras ASCII, números, _ o -.")
    return name


@dataclass
class AudioArtifact:
    """Audio con frecuencia explícita; el array pertenece a esta etapa."""

    audio: np.ndarray
    sr: int

    def __post_init__(self):
        self.audio = np.array(self.audio, dtype=np.float32, copy=True)
        if not isinstance(self.sr, (int, np.integer)) or self.sr <= 0:
            raise ValueError("sr debe ser un entero positivo en Hz.")
        if self.audio.ndim not in (1, 2) or self.audio.size == 0:
            raise ValueError("Se espera audio no vacío mono o (canales, muestras).")
        if self.audio.ndim == 2 and self.audio.shape[0] not in (1, 2):
            raise ValueError("Solo se acepta mono o estéreo, con canales en el primer eje.")
        if not np.isfinite(self.audio).all():
            raise ValueError("El audio contiene NaN o infinito.")

    @property
    def duration(self) -> float:
        return self.audio.shape[-1] / self.sr

    def mono(self, sr: int = 22050) -> AudioArtifact:
        audio = self.audio.mean(axis=0) if self.audio.ndim == 2 else self.audio
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=self.sr, target_sr=sr)
        return AudioArtifact(audio, sr)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # FLOAT evita clipping o normalizaciones independientes entre etapas.
        sf.write(path, self.audio.T, self.sr, subtype="FLOAT")
        return path


@dataclass
class SeparationResult:
    model_name: str
    input_audio: AudioArtifact
    demucs_input: AudioArtifact
    stems: Dict[str, AudioArtifact]


@dataclass
class DSPVariant:
    """Una rama independiente con etapas consecutivas y post-DSP opcional.

    Cada AudioPreProcessor es una etapa recuperable. Para observar cada efecto,
    usar un procesador por efecto en vez de un único Pedalboard con todos ellos.
    Las transformaciones deben conservar frecuencia, duración y forma mono.
    """

    name: str
    stages: Sequence[AudioPreProcessor] = field(default_factory=tuple)
    note_postprocessor: Optional[NotePostProcessor] = None

    def __post_init__(self):
        _identifier(self.name)
        names = [_identifier(stage.name) for stage in self.stages]
        if len(set(names)) != len(names) or "input_22050" in names:
            raise ValueError("Cada etapa necesita un nombre único y no reservado.")


@dataclass
class VariantResult:
    stem: str
    name: str
    stages: Dict[str, AudioArtifact]
    note_postprocessor: Optional[NotePostProcessor] = None
    transcription: Optional[dict] = None

    @property
    def final_audio(self) -> AudioArtifact:
        return next(reversed(self.stages.values()))


class DemucsTranscriber:
    """Coordinador reutilizable. No carga modelos hasta que son necesarios.

    Ejemplo::

        engine = DemucsTranscriber()
        separated = engine.separate("mix.wav")
        variants = engine.process_variants(separated, {
            "bass": [DSPVariant("raw"), DSPVariant("filtered", [preprocessor])],
        })
        results = engine.transcribe_variants(variants)
        engine.export(separated, results, "outputs/experiment")

    CPU es el valor predeterminado portable; device="mps" usa GPU de Apple
    Silicon y device="cuda" usa GPU NVIDIA, cuando están disponibles.
    La primera separación descarga pesos mediante la librería Demucs.
    """

    def __init__(self, model_name: str = "htdemucs", device: str = "cpu",
                 shifts: int = 0, overlap: float = 0.25,
                 transcriber: Optional[BasicPitchTranscriber] = None):
        if model_name != "htdemucs":
            raise ValueError("Esta integración utiliza el modelo htdemucs de cuatro stems.")
        if shifts < 0 or not 0 <= overlap < 1:
            raise ValueError("shifts debe ser >= 0 y overlap debe estar en [0, 1).")
        if device not in available_devices():
            raise ValueError(f"El dispositivo '{device}' no está disponible en este proceso. "
                             "Usa 'cpu' o consulta available_devices().")
        self.model_name = model_name
        self.device = device
        self.shifts = shifts
        self.overlap = overlap
        self.transcriber = transcriber if transcriber is not None else BasicPitchTranscriber()
        self._separator_model = None

    @property
    def separator_model(self):
        if self._separator_model is None:
            from demucs.pretrained import get_model

            model = get_model(self.model_name)
            model.eval()
            self._separator_model = model
        return self._separator_model

    def separate(self, audio_input: Union[str, Path, np.ndarray],
                 sr: Optional[int] = None) -> SeparationResult:
        """Separa una ruta o array. Para arrays, sr es obligatorio."""
        if isinstance(audio_input, (str, Path)):
            audio, input_sr = librosa.load(str(audio_input), sr=None, mono=False)
        else:
            if sr is None:
                raise ValueError("Debe indicarse sr para entradas numpy.")
            audio, input_sr = audio_input, sr
        original = AudioArtifact(audio, input_sr)
        model = self.separator_model
        waveform = original.audio
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        if model.audio_channels == 2 and waveform.shape[0] == 1:
            waveform = np.repeat(waveform, 2, axis=0)
        if waveform.shape[0] != model.audio_channels:
            raise ValueError("Número de canales incompatible con Demucs.")
        if original.sr != model.samplerate:
            waveform = librosa.resample(waveform, orig_sr=original.sr,
                                       target_sr=model.samplerate, axis=-1)
        prepared = AudioArtifact(waveform, model.samplerate)

        import torch
        from demucs.apply import apply_model

        wav = torch.from_numpy(prepared.audio)
        # Misma normalización que la interfaz oficial; estable para silencio.
        reference = wav.mean(0)
        mean = reference.mean()
        std = reference.std(unbiased=False).clamp_min(1e-8)
        with torch.inference_mode():
            output = apply_model(model, ((wav - mean) / std)[None],
                                 device=self.device, shifts=self.shifts,
                                 split=True, overlap=self.overlap, progress=False)
        output = (output[0].cpu() * std + mean).numpy()
        stems = {name: AudioArtifact(samples, model.samplerate)
                 for name, samples in zip(model.sources, output)}
        return SeparationResult(self.model_name, original, prepared, stems)

    def process_variants(self, separation: SeparationResult,
                         pipelines: Mapping[str, Sequence[DSPVariant]]) -> Dict[str, Dict[str, VariantResult]]:
        """DSP sin inferencia: cada rama parte de una copia del stem mono a 22.05 kHz."""
        unknown = set(pipelines) - set(separation.stems)
        if unknown:
            raise ValueError(f"Stems desconocidos: {sorted(unknown)}")
        results = {}
        for stem, variants in pipelines.items():
            names = [variant.name for variant in variants]
            if len(set(names)) != len(names):
                raise ValueError(f"Nombres de variantes duplicados en {stem}.")
            base = separation.stems[stem].mono(22050)
            results[stem] = {}
            for variant in variants:
                stages = {"input_22050": AudioArtifact(base.audio, base.sr)}
                current = stages["input_22050"]
                for processor in variant.stages:
                    processed = processor.process(current.audio.copy(), current.sr)
                    artifact = AudioArtifact(processed, current.sr)
                    if artifact.audio.shape != current.audio.shape:
                        raise ValueError(f"La etapa {processor.name} debe conservar la forma del audio.")
                    stages[processor.name] = artifact
                    current = artifact
                results[stem][variant.name] = VariantResult(
                    stem, variant.name, stages, variant.note_postprocessor)
        return results

    def transcribe_variants(self, variants: Mapping[str, Mapping[str, VariantResult]]) -> Dict[str, Dict[str, VariantResult]]:
        """Transcribe solo las ramas recibidas, reutilizando Basic Pitch.

        No modifica los resultados DSP de entrada; conserva MIDI crudo y filtrado.
        Drums puede procesarse/exportarse sin incluirlo en esta llamada.
        """
        transcribed = {}
        for stem, branches in variants.items():
            transcribed[stem] = {}
            for name, result in branches.items():
                transcription = self.transcriber.transcribe(
                    result.final_audio.audio, sr=result.final_audio.sr,
                    note_postprocessor=result.note_postprocessor)
                transcription["pipeline_name"] = result.name
                transcription["applied_pre_dsp"] = len(result.stages) > 1
                transcribed[stem][name] = replace(result, transcription=transcription)
        return transcribed

    @staticmethod
    def export(separation: SeparationResult,
               variants: Mapping[str, Mapping[str, VariantResult]],
               output_dir: Union[str, Path]) -> Path:
        """Exporta WAV FLOAT, MIDI y manifest.json. Usar una carpeta por experimento."""
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {"model": separation.model_name, "audio": {}, "transcriptions": []}

        def save_audio(artifact, relative):
            artifact.save(root / relative)
            manifest["audio"][relative] = {"sr": int(artifact.sr),
                "channels": 1 if artifact.audio.ndim == 1 else artifact.audio.shape[0],
                "duration_seconds": artifact.duration}

        save_audio(separation.input_audio, "input.wav")
        save_audio(separation.demucs_input, "demucs_input.wav")
        for stem, artifact in separation.stems.items():
            save_audio(artifact, f"stems/{_identifier(stem)}.wav")
        for stem, branches in variants.items():
            for name, result in branches.items():
                prefix = f"variants/{_identifier(stem)}/{_identifier(name)}"
                for index, (stage, artifact) in enumerate(result.stages.items()):
                    save_audio(artifact, f"{prefix}/{index:02d}_{_identifier(stage)}.wav")
                if result.transcription is not None:
                    for key in ("midi", "raw_midi"):
                        result.transcription[key].write(str(root / prefix / f"{key}.mid"))
                    manifest["transcriptions"].append({"stem": stem, "variant": name,
                        "raw_notes": len(result.transcription["raw_note_events"]),
                        "final_notes": len(result.transcription["note_events"])})
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return root
