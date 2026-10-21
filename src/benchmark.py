"""Métricas de transcripción del notebook 06, sin ejecutar modelos.

Las predicciones usan tiempos relativos al fragmento. Las referencias originales
deben recortarse con ``load_reference_midis`` antes de compararlas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, TYPE_CHECKING, Union

import mir_eval
import numpy as np
import pretty_midi

if TYPE_CHECKING:
    from .demucs_transcriber import VariantResult

ONSET_TOL = 0.05
PITCH_TOL_CENTS = 50.0
FRAME_HOP_SEC = 0.01
METRIC_KEYS = ("P_no", "R_no", "F_no", "P", "R", "F", "Acc")
BENCHMARK_COLUMNS = (
    "Variante", "Salida", "F_no", "F", "Acc", "Δ F_no", "Δ F", "Δ Acc",
    "Notas GT", "Notas predichas", "P_no", "R_no", "P", "R", "estado", "Stem",
)


def _validate_window(start: float, duration: float) -> None:
    if not np.isfinite(start) or start < 0:
        raise ValueError("El inicio debe ser finito y no negativo.")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("La duración debe ser finita y positiva.")


def crop_midi(midi: pretty_midi.PrettyMIDI, start: float,
              duration: float) -> pretty_midi.PrettyMIDI:
    """Copia notas no percusivas en [start, start+duration), relativas a cero.

    Recorta notas sostenidas en los bordes; no altera el MIDI original.
    Esta copia para evaluación contiene notas, no automatizaciones ni pitch bends.
    """
    _validate_window(start, duration)
    cropped = pretty_midi.PrettyMIDI()
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        target = pretty_midi.Instrument(program=instrument.program, name=instrument.name)
        for note in instrument.notes:
            onset = max(note.start, start)
            offset = min(note.end, start + duration)
            if offset > onset:
                target.notes.append(pretty_midi.Note(
                    velocity=note.velocity, pitch=note.pitch,
                    start=onset - start, end=offset - start))
        if target.notes:
            cropped.instruments.append(target)
    return cropped


def load_reference_midis(paths: Sequence[Union[str, Path]], start: float,
                         duration: float) -> pretty_midi.PrettyMIDI:
    """Combina los MIDI seleccionados y recorta su ventana absoluta una vez."""
    _validate_window(start, duration)
    reference = pretty_midi.PrettyMIDI()
    for path in dict.fromkeys(Path(path).resolve() for path in paths):
        cropped = crop_midi(pretty_midi.PrettyMIDI(str(path)), start, duration)
        reference.instruments.extend(cropped.instruments)
    return reference


def midi_to_intervals_and_pitches(midi: pretty_midi.PrettyMIDI):
    """Intervalos en segundos y alturas en Hz, como en el notebook 06."""
    notes = sorted((note.start, note.end, note.pitch)
                   for instrument in midi.instruments if not instrument.is_drum
                   for note in instrument.notes)
    if not notes:
        return np.empty((0, 2)), np.empty(0)
    return (np.array([(start, end) for start, end, _ in notes], dtype=float),
            np.array([pretty_midi.note_number_to_hz(pitch) for _, _, pitch in notes]))


def midi_to_frame_frequencies(midi: pretty_midi.PrettyMIDI, total_duration: float,
                              hop_sec: float = FRAME_HOP_SEC):
    """Alturas activas únicas por frame: dos instrumentos al unísono cuentan una vez."""
    _validate_window(0, total_duration)
    if not np.isfinite(hop_sec) or hop_sec <= 0:
        raise ValueError("El salto de evaluación debe ser finito y positivo.")
    times = np.arange(int(np.ceil(total_duration / hop_sec))) * hop_sec
    active = [set() for _ in times]
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            first = np.searchsorted(times, note.start, side="left")
            last = np.searchsorted(times, note.end, side="left")
            for frame in active[first:last]:
                frame.add(note.pitch)
    return [np.array([pretty_midi.note_number_to_hz(pitch) for pitch in sorted(frame)])
            for frame in active]


def evaluate_prediction(gt_midi: pretty_midi.PrettyMIDI,
                        pred_midi: pretty_midi.PrettyMIDI,
                        total_duration: float) -> dict:
    """Evalúa notas y frames con las tolerancias del 06.

    Referencia vacía: métricas None y estado explícito. Predicción vacía con
    referencia: ceros. Ambas entradas se limitan a la duración real del audio.
    """
    reference = crop_midi(gt_midi, 0, total_duration)
    prediction = crop_midi(pred_midi, 0, total_duration)
    ref_int, ref_pitch = midi_to_intervals_and_pitches(reference)
    est_int, est_pitch = midi_to_intervals_and_pitches(prediction)
    counts = {"Notas GT": len(ref_int), "Notas predichas": len(est_int)}
    if not len(ref_int):
        return {**counts, "estado": "sin referencia evaluable",
                **dict.fromkeys(METRIC_KEYS)}
    if not len(est_int):
        return {**counts, "estado": "ok", **dict.fromkeys(METRIC_KEYS, 0.0)}

    kwargs = dict(onset_tolerance=ONSET_TOL, pitch_tolerance=PITCH_TOL_CENTS)
    p_no, r_no, f_no, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pitch, est_int, est_pitch, offset_ratio=None, **kwargs)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pitch, est_int, est_pitch,
        offset_ratio=0.2, offset_min_tolerance=0.05, **kwargs)
    times = np.arange(int(np.ceil(total_duration / FRAME_HOP_SEC))) * FRAME_HOP_SEC
    ref_frames = midi_to_frame_frequencies(reference, total_duration)
    est_frames = midi_to_frame_frequencies(prediction, total_duration)
    accuracy = mir_eval.multipitch.metrics(times, ref_frames, times, est_frames, window=0.5)[2]
    return {**counts, "estado": "ok", "P_no": float(p_no), "R_no": float(r_no),
            "F_no": float(f_no), "P": float(p), "R": float(r), "F": float(f),
            "Acc": float(accuracy)}


def benchmark_variants(variants: Mapping[str, VariantResult],
                       reference: pretty_midi.PrettyMIDI,
                       total_duration: float) -> list[dict]:
    """Tabla por variante/salida; deltas contra el MIDI del modelo de ``raw``.

    No carga ni ejecuta Demucs o Basic Pitch. No evalúa percusión.
    """
    for result in variants.values():
        if result.stem == "drums":
            raise ValueError("El benchmark de notas excluye drums.")
        if result.transcription is None:
            raise ValueError("Transcribe las variantes antes de evaluar.")
    baseline = None
    if "raw" in variants:
        baseline = evaluate_prediction(reference, variants["raw"].transcription["raw_midi"], total_duration)
    rows = []
    for name, result in variants.items():
        outputs = [("modelo", "raw_midi")]
        if result.note_postprocessor is not None:
            outputs.append(("post-DSP", "midi"))
        for output_name, key in outputs:
            metrics = evaluate_prediction(reference, result.transcription[key], total_duration)
            row = {"Stem": result.stem, "Variante": name, "Salida": output_name, **metrics}
            for metric in ("F_no", "F", "Acc"):
                value = metrics[metric]
                base_value = baseline[metric] if baseline is not None else None
                row[f"Δ {metric}"] = (value - base_value
                                       if value is not None and base_value is not None else None)
            rows.append({key: row[key] for key in BENCHMARK_COLUMNS})
    return rows
