"""
Módulo de Transcripción y Procesamiento de Señales (DSP) para AcusPy.

Proporciona:
- AudioPreProcessor: Pipeline de pre-procesamiento acústico (DSP sobre audio con Pedalboard).
- NotePostProcessor: Pipeline de post-procesamiento simbólico (Filtrado de notas MIDI).
- TranscriptionPipeline: Contenedor que desacopla y unifica Pre-DSP y Post-DSP.
- BasicPitchTranscriber: Clase envoltorio del modelo Basic Pitch que gestiona inferencia y pipelines.
- create_instrument_pipeline: Fábrica para instanciar pipelines adaptados a instrumentos.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from pedalboard import HighpassFilter, LowpassFilter, NoiseGate, Pedalboard


# ==============================================================================
# 1. PRE-PROCESAMIENTO DE AUDIO (Audio DSP)
# ==============================================================================

class AudioPreProcessor:
    """
    Pipeline de pre-procesamiento acústico que opera directamente sobre la señal de audio
    antes de ser enviada al modelo de IA.
    
    Acepta efectos de Pedalboard o funciones personalizadas (audio, sr) -> audio.
    """
    def __init__(
        self,
        board: Optional[Union[Pedalboard, list]] = None,
        custom_transforms: Optional[List[Callable[[np.ndarray, int], np.ndarray]]] = None,
        name: str = "AudioPreProcessor",
    ):
        """
        Args:
            board: Objeto Pedalboard o lista de plugins de Pedalboard (Highpass, NoiseGate, etc.).
            custom_transforms: Lista de funciones con firma (audio, sr) -> audio.
            name: Nombre identificador del procesador.
        """
        self.name = name
        if isinstance(board, list):
            self.board = Pedalboard(board)
        elif isinstance(board, Pedalboard):
            self.board = board
        else:
            self.board = None

        self.custom_transforms = custom_transforms or []

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Aplica la cadena de efectos de audio.

        Args:
            audio: Arreglo de audio (mono, float32 o float64).
            sr: Frecuencia de muestreo en Hz.

        Returns:
            np.ndarray: Señal de audio procesada.
        """
        y = audio.astype(np.float32)

        # 1. Aplicar Pedalboard si está definido
        if self.board is not None and len(self.board) > 0:
            y = self.board(y, sr)

        # 2. Aplicar transformaciones personalizadas
        for transform in self.custom_transforms:
            y = transform(y, sr)

        return y

    def __repr__(self) -> str:
        plugins = [p.__class__.__name__ for p in self.board] if self.board else []
        return f"AudioPreProcessor(name='{self.name}', plugins={plugins}, custom_transforms={len(self.custom_transforms)})"


# ==============================================================================
# 2. POST-PROCESAMIENTO SIMBÓLICO (Note / MIDI DSP)
# ==============================================================================

class NotePostProcessor:
    """
    Pipeline de post-procesamiento simbólico que opera sobre los eventos de notas
    y/o el archivo MIDI resultante de Basic Pitch.
    
    Aplica filtros de reglas acústicas/musicales (duración, tesitura, velocidad).
    """
    def __init__(
        self,
        pitch_range: Optional[Tuple[int, int]] = None,
        min_duration_sec: Optional[float] = None,
        min_velocity: Optional[float] = None,
        custom_filter: Optional[Callable[[dict], bool]] = None,
        name: str = "NotePostProcessor",
    ):
        """
        Args:
            pitch_range: Tupla (pitch_min, pitch_max) en números MIDI (ej. [28, 67] para bajo).
            min_duration_sec: Duración mínima en segundos para descartar notas fantasma (ej. 0.05).
            min_velocity: Velocidad o confianza mínima para descartar notas tenues (0 a 127).
            custom_filter: Función que recibe un dict de nota {'onset', 'offset', 'pitch', 'velocity'} 
                           y devuelve True si debe conservarse.
            name: Nombre identificador del post-procesador.
        """
        self.name = name
        self.pitch_range = pitch_range
        self.min_duration_sec = min_duration_sec
        self.min_velocity = min_velocity
        self.custom_filter = custom_filter

    def process(
        self,
        note_events: List[Tuple[float, float, int, float, Optional[List[int]]]],
        base_midi: Optional[pretty_midi.PrettyMIDI] = None,
    ) -> Tuple[List[Tuple[float, float, int, float, Optional[List[int]]]], pretty_midi.PrettyMIDI]:
        """
        Filtra la lista de eventos de notas y reconstruye un nuevo PrettyMIDI limpio.

        Args:
            note_events: Lista de tuplas (onset, offset, pitch, velocity, [pitch_bends]).
            base_midi: Objeto PrettyMIDI de referencia para copiar metadatos de instrumento/tempo.

        Returns:
            Tupla (filtered_note_events, filtered_midi).
        """
        filtered_events = []

        for event in note_events:
            onset, offset, pitch, velocity = event[0], event[1], event[2], event[3]
            duration = offset - onset

            # Filtro 1: Duración mínima
            if self.min_duration_sec is not None and duration < self.min_duration_sec:
                continue

            # Filtro 2: Rango de pitch (tesitura)
            if self.pitch_range is not None:
                p_min, p_max = self.pitch_range
                if not (p_min <= pitch <= p_max):
                    continue

            # Filtro 3: Velocidad mínima
            if self.min_velocity is not None and velocity < self.min_velocity:
                continue

            # Filtro 4: Función personalizada
            if self.custom_filter is not None:
                note_dict = {
                    "onset": onset,
                    "offset": offset,
                    "pitch": pitch,
                    "velocity": velocity,
                }
                if not self.custom_filter(note_dict):
                    continue

            filtered_events.append(event)

        # Reconstruir PrettyMIDI con las notas filtradas
        new_midi = pretty_midi.PrettyMIDI()
        
        # Heredar programa de instrumento si está disponible en base_midi
        program = 0
        is_drum = False
        name = "Filtered Instrument"
        if base_midi and len(base_midi.instruments) > 0:
            program = base_midi.instruments[0].program
            is_drum = base_midi.instruments[0].is_drum
            name = base_midi.instruments[0].name

        instrument = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
        for event in filtered_events:
            note = pretty_midi.Note(
                velocity=int(event[3]),
                pitch=int(event[2]),
                start=float(event[0]),
                end=float(event[1]),
            )
            instrument.notes.append(note)

        new_midi.instruments.append(instrument)

        return filtered_events, new_midi

    def __repr__(self) -> str:
        return (
            f"NotePostProcessor(name='{self.name}', pitch_range={self.pitch_range}, "
            f"min_duration_sec={self.min_duration_sec}, min_velocity={self.min_velocity})"
        )


# ==============================================================================
# 3. CONTENEDOR UNIFICADO: TranscriptionPipeline
# ==============================================================================

class TranscriptionPipeline:
    """
    Contenedor que desacopla y orquesta un AudioPreProcessor (Pre-DSP)
    y un NotePostProcessor (Post-DSP).
    """
    def __init__(
        self,
        name: str = "CustomPipeline",
        audio_preprocessor: Optional[AudioPreProcessor] = None,
        note_postprocessor: Optional[NotePostProcessor] = None,
    ):
        self.name = name
        self.audio_preprocessor = audio_preprocessor
        self.note_postprocessor = note_postprocessor

    @property
    def has_audio_dsp(self) -> bool:
        return self.audio_preprocessor is not None

    @property
    def has_note_dsp(self) -> bool:
        return self.note_postprocessor is not None

    def __repr__(self) -> str:
        return (
            f"TranscriptionPipeline(name='{self.name}', "
            f"pre_dsp={bool(self.audio_preprocessor)}, "
            f"post_dsp={bool(self.note_postprocessor)})"
        )


# ==============================================================================
# 4. FÁBRICA AUTOMÁTICA DE PIPELINES POR INSTRUMENTO
# ==============================================================================

def get_audio_pipeline(inst_class: str) -> Optional[AudioPreProcessor]:
    """
    Crea automáticamente un AudioPreProcessor con parámetros acústicos
    adaptados a la clase de instrumento.
    """
    cls_lower = inst_class.lower()

    if "bass" in cls_lower:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=35.0),
            LowpassFilter(cutoff_frequency_hz=1800.0),
            NoiseGate(threshold_db=-38.0, ratio=8.0),
        ])
        return AudioPreProcessor(board=board, name=f"{inst_class}_PreDSP")

    elif "guitar" in cls_lower:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80.0),
            LowpassFilter(cutoff_frequency_hz=5000.0),
            NoiseGate(threshold_db=-42.0, ratio=6.0),
        ])
        return AudioPreProcessor(board=board, name=f"{inst_class}_PreDSP")

    elif "piano" in cls_lower:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=27.5),
            LowpassFilter(cutoff_frequency_hz=8000.0),
        ])
        return AudioPreProcessor(board=board, name=f"{inst_class}_PreDSP")

    elif "organ" in cls_lower:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=40.0),
            LowpassFilter(cutoff_frequency_hz=6000.0),
        ])
        return AudioPreProcessor(board=board, name=f"{inst_class}_PreDSP")

    elif "strings" in cls_lower:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=65.0),
            LowpassFilter(cutoff_frequency_hz=7000.0),
        ])
        return AudioPreProcessor(board=board, name=f"{inst_class}_PreDSP")

    # Por defecto para instrumentos desconocidos
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30.0),
    ])
    return AudioPreProcessor(board=board, name=f"{inst_class}_DefaultPreDSP")


def get_note_pipeline(inst_class: str) -> Optional[NotePostProcessor]:
    """
    Crea automáticamente un NotePostProcessor con tesitura y umbrales
    adaptados a la clase de instrumento.
    """
    cls_lower = inst_class.lower()

    if "bass" in cls_lower:
        return NotePostProcessor(
            pitch_range=(28, 67),       # E1 a G4
            min_duration_sec=0.05,      # 50 ms
            name=f"{inst_class}_PostDSP",
        )

    elif "guitar" in cls_lower:
        return NotePostProcessor(
            pitch_range=(40, 88),       # E2 a E6
            min_duration_sec=0.04,      # 40 ms
            name=f"{inst_class}_PostDSP",
        )

    elif "piano" in cls_lower:
        return NotePostProcessor(
            pitch_range=(21, 108),      # A0 a C8
            min_duration_sec=0.03,      # 30 ms
            name=f"{inst_class}_PostDSP",
        )

    elif "organ" in cls_lower:
        return NotePostProcessor(
            pitch_range=(36, 96),       # C2 a C7
            min_duration_sec=0.04,
            name=f"{inst_class}_PostDSP",
        )

    elif "strings" in cls_lower:
        return NotePostProcessor(
            pitch_range=(36, 100),      # C2 a E7
            min_duration_sec=0.05,
            name=f"{inst_class}_PostDSP",
        )

    # Por defecto
    return NotePostProcessor(
        min_duration_sec=0.04,
        name=f"{inst_class}_DefaultPostDSP",
    )


def create_instrument_pipeline(
    inst_class: str,
    enable_pre: bool = True,
    enable_post: bool = True,
    custom_name: Optional[str] = None,
) -> TranscriptionPipeline:
    """
    Fábrica de conveniencia que crea un TranscriptionPipeline completo
    a partir del nombre o clase del instrumento (obtenido típicamente de metadata.yaml).

    Args:
        inst_class: Clase del instrumento (ej. 'Bass', 'Guitar', 'Piano').
        enable_pre: Si es True, incluye el AudioPreProcessor acústico.
        enable_post: Si es True, incluye el NotePostProcessor simbólico.
        custom_name: Nombre personalizado para el pipeline.

    Returns:
        TranscriptionPipeline configurado.
    """
    name = custom_name or f"Pipeline_{inst_class}"
    audio_pre = get_audio_pipeline(inst_class) if enable_pre else None
    note_post = get_note_pipeline(inst_class) if enable_post else None

    return TranscriptionPipeline(
        name=name,
        audio_preprocessor=audio_pre,
        note_postprocessor=note_post,
    )


# ==============================================================================
# 5. WRAPPER DEL MODELO: BasicPitchTranscriber
# ==============================================================================

class BasicPitchTranscriber:
    """
    Envoltorio del modelo Basic Pitch que mantiene el modelo compilado en memoria
    y desacopla la ejecución de AudioPreProcessor y NotePostProcessor.
    """
    def __init__(self, model_or_path=ICASSP_2022_MODEL_PATH):
        """
        Inicializa y carga los pesos del modelo Basic Pitch una sola vez.
        """
        self.model_path = model_or_path
        self.sample_rate = 22050  # Frecuencia nativa de Basic Pitch
        print(f"📦 [BasicPitchTranscriber] Modelo cargado: {self.model_path}")

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sr: int = 22050,
        pipeline: Optional[TranscriptionPipeline] = None,
        audio_preprocessor: Optional[AudioPreProcessor] = None,
        note_postprocessor: Optional[NotePostProcessor] = None,
    ) -> Dict[str, any]:
        """
        Ejecuta el flujo completo de transcripción:
        1. Carga / normaliza el audio a 22050 Hz.
        2. Aplica Pre-DSP de Audio (si está configurado).
        3. Realiza la inferencia con Basic Pitch.
        4. Aplica Post-DSP de Notas (si está configurado).
        5. Devuelve un diccionario estructurado listo para mir_eval.

        Args:
            audio_input: Ruta al archivo de audio o arreglo numpy con la señal mono.
            sr: Sample rate si audio_input es un numpy array (default 22050).
            pipeline: Instancia de TranscriptionPipeline opcional.
            audio_preprocessor: Preprocesador de audio explícito (sobrescribe pipeline si se provee).
            note_postprocessor: Postprocesador de notas explícito (sobrescribe pipeline si se provee).

        Returns:
            Dict con:
                - 'midi': PrettyMIDI final (con filtros aplicados si corresponde).
                - 'note_events': Lista de eventos de notas finales (onset, offset, pitch, velocity).
                - 'audio_processed': Señal de audio procesada por Pre-DSP.
                - 'sr': Frecuencia de muestreo (22050 Hz).
                - 'raw_midi': PrettyMIDI crudo del modelo.
                - 'raw_note_events': Eventos crudos del modelo.
                - 'model_output': Diccionario con activaciones de Basic Pitch ('contour', 'onset', 'note').
                - 'pipeline_name': Nombre del pipeline ejecutado.
        """
        # Resolver procesadores a usar
        pre_proc = audio_preprocessor or (pipeline.audio_preprocessor if pipeline else None)
        post_proc = note_postprocessor or (pipeline.note_postprocessor if pipeline else None)
        pipeline_name = pipeline.name if pipeline else ("Raw" if (not pre_proc and not post_proc) else "Custom")

        # 1. Cargar señal de audio a 22050 Hz
        if isinstance(audio_input, (str, Path)):
            audio_path = str(audio_input)
            audio_signal, loaded_sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        else:
            if sr != self.sample_rate:
                audio_signal = librosa.resample(audio_input, orig_sr=sr, target_sr=self.sample_rate)
            else:
                audio_signal = audio_input.copy()
            audio_path = None

        # 2. Pre-procesamiento de Audio (Pre-DSP)
        if pre_proc is not None:
            processed_audio = pre_proc.process(audio_signal, self.sample_rate)
        else:
            processed_audio = audio_signal

        # 3. Inferencia con Basic Pitch
        # Si hubo pre-procesamiento de audio o la entrada era un np.ndarray,
        # escribimos un WAV temporal para Basic Pitch predict()
        temp_wav_path = None
        try:
            if pre_proc is not None or audio_path is None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_wav_path = tmp.name
                sf.write(temp_wav_path, processed_audio, self.sample_rate)
                inference_audio_path = temp_wav_path
            else:
                inference_audio_path = audio_path

            model_output, raw_midi, raw_note_events = predict(
                inference_audio_path,
                model_or_model_path=self.model_path,
            )
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

        # 4. Post-procesamiento de Notas (Post-DSP)
        if post_proc is not None:
            final_note_events, final_midi = post_proc.process(raw_note_events, base_midi=raw_midi)
        else:
            final_note_events = raw_note_events
            final_midi = raw_midi

        return {
            "midi": final_midi,
            "note_events": final_note_events,
            "audio_processed": processed_audio,
            "sr": self.sample_rate,
            "raw_midi": raw_midi,
            "raw_note_events": raw_note_events,
            "model_output": model_output,
            "pipeline_name": pipeline_name,
            "applied_pre_dsp": pre_proc is not None,
            "applied_post_dsp": post_proc is not None,
        }
