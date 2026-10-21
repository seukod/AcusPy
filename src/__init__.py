"""
Paquete AcusPy (src).
Exporta componentes principales de transcripción y DSP.
"""

from .transcriber import (
    AudioPreProcessor,
    BasicPitchTranscriber,
    NotePostProcessor,
    TranscriptionPipeline,
    create_instrument_pipeline,
    get_audio_pipeline,
    get_note_pipeline,
)
from .demucs_transcriber import (
    AudioArtifact, DemucsTranscriber, DSPVariant, SeparationResult, VariantResult,
    available_devices,
)

__all__ = [
    "AudioArtifact",
    "available_devices",
    "DemucsTranscriber",
    "DSPVariant",
    "SeparationResult",
    "VariantResult",
    "AudioPreProcessor",
    "BasicPitchTranscriber",
    "NotePostProcessor",
    "TranscriptionPipeline",
    "create_instrument_pipeline",
    "get_audio_pipeline",
    "get_note_pipeline",
]
