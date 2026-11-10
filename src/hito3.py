"""Contrato común del Hito 3: catálogo de tracks, condiciones 2×2 y formato de resultados.

Todos los notebooks de ``notebooks/hito_03`` leen y escriben con estas funciones, para
que los CSV de distintas personas y máquinas (local o Patagón) sean comparables.

Unidad de evaluación: un track × un grupo Demucs (``bass``, ``guitar``…). La referencia es
la unión de los MIDI de todos los instrumentos de ese track que caen en ese grupo, porque
Demucs entrega un solo stem por grupo: si un track tiene dos bajos, el stem ``bass`` trae los dos.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "babyslakh_16k"
RESULTS_DIR = PROJECT_ROOT / "results" / "hito3"
SR_TRANSCRIPTION = 22050

# (clave, etiqueta, pre_dsp, post_dsp) en el orden en que se reportan.
CONDITIONS = (
    ("raw", "Sin DSP", False, False),
    ("pre", "Solo PRE", True, False),
    ("post", "Solo POST", False, True),
    ("prepost", "PRE + POST", True, True),
)

METRICS = ("P_no", "R_no", "F_no", "P", "R", "F", "Acc")

RESULT_COLUMNS = (
    "track_id", "grupo", "stems_ref", "inst_class", "modelo_demucs",
    "condicion", "pre_dsp", "post_dsp", "inicio_s", "duracion_s", "ref_transpose",
    *METRICS, "n_ref", "n_pred", "estado",
)

CATALOG_COLUMNS = ("track_id", "grupo", "stems_ref", "inst_class", "programas", "evaluable")

# Grupos que entrega cada modelo de Demucs.
MODEL_GROUPS = {
    "htdemucs": ("drums", "bass", "other", "vocals"),
    "htdemucs_6s": ("drums", "bass", "other", "vocals", "guitar", "piano"),
}


def demucs_group(inst_class: str, program_name: str = "", model: str = "htdemucs") -> str:
    """Grupo Demucs donde cae un instrumento de BabySlakh.

    Misma regla que el notebook 09, más guitarra y piano cuando el modelo los separa.
    """
    cls, prog = inst_class.lower(), program_name.lower()
    if "drum" in cls:
        return "drums"
    if "bass" in cls:
        return "bass"
    if "choir" in prog or "voice" in prog or "vocal" in cls:
        return "vocals"
    groups = MODEL_GROUPS[model]
    if "guitar" in groups and "guitar" in cls:
        return "guitar"
    if "piano" in groups and "piano" in cls:
        return "piano"
    return "other"


def reference_transpose(group: str) -> int:
    """Semitonos que se aplican a la referencia antes de evaluar.

    El render de BabySlakh del bajo suena una octava bajo su MIDI (medido en el Hito 2,
    9 de 10 tracks): se calibra la referencia, no el modelo.
    """
    return -12 if group == "bass" else 0


@dataclass
class TrackGroup:
    """Un track × un grupo Demucs, con los MIDI que forman su referencia."""
    track_id: str
    track_dir: Path
    group: str
    stem_ids: list[str] = field(default_factory=list)
    inst_classes: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    missing_midi: list[str] = field(default_factory=list)

    @property
    def mix_path(self) -> Path:
        return self.track_dir / "mix.wav"

    @property
    def midi_paths(self) -> list[Path]:
        return [self.track_dir / "MIDI" / f"{s}.mid" for s in self.stem_ids]

    @property
    def evaluable(self) -> bool:
        """Solo si todos los instrumentos del grupo tienen MIDI: una referencia parcial
        contaría como fantasmas las notas correctas del instrumento faltante."""
        return bool(self.stem_ids) and not self.missing_midi

    def catalog_row(self) -> dict:
        return {
            "track_id": self.track_id, "grupo": self.group,
            "stems_ref": "+".join(self.stem_ids + self.missing_midi),
            "inst_class": "+".join(sorted(set(self.inst_classes))),
            "programas": "; ".join(self.programs), "evaluable": int(self.evaluable),
        }


def list_track_groups(group: str = "bass", data_dir: Union[str, Path] = DATA_DIR,
                      model: str = "htdemucs") -> list[TrackGroup]:
    """Todos los tracks que tienen al menos un instrumento del grupo, evaluables o no."""
    if group not in MODEL_GROUPS[model] or group == "drums":
        raise ValueError(f"El grupo {group!r} no es tonal o no lo separa {model}.")
    out = []
    for track_dir in sorted(Path(data_dir).iterdir()):
        meta_path = track_dir / "metadata.yaml"
        if not meta_path.is_file():
            continue
        stems = (yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}).get("stems") or {}
        tg = TrackGroup(track_dir.name, track_dir, group)
        for stem_id, info in sorted(stems.items()):
            if info.get("is_drum"):
                continue
            cls, prog = info.get("inst_class", ""), info.get("midi_program_name", "")
            if demucs_group(cls, prog, model) != group:
                continue
            target = tg.stem_ids if (track_dir / "MIDI" / f"{stem_id}.mid").is_file() else tg.missing_midi
            target.append(stem_id)
            tg.inst_classes.append(cls)
            tg.programs.append(prog)
        if tg.stem_ids or tg.missing_midi:
            out.append(tg)
    return out


def write_rows(rows: Iterable[dict], path: Union[str, Path],
               columns: Sequence[str] = RESULT_COLUMNS) -> Path:
    """Escribe filas con columnas fijas; falla si una fila trae columnas fuera del contrato."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    for row in rows:
        extra = set(row) - set(columns)
        if extra:
            raise ValueError(f"Columnas fuera del contrato: {sorted(extra)}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return path


def read_rows(path: Union[str, Path]) -> list[dict]:
    """Lee un CSV del contrato convirtiendo métricas y conteos a número (vacío → None)."""
    numeric = set(METRICS) | {"inicio_s", "duracion_s"}
    integer = {"pre_dsp", "post_dsp", "ref_transpose", "n_ref", "n_pred", "evaluable"}
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                if k in numeric:
                    row[k] = float(v) if v != "" else None
                elif k in integer:
                    row[k] = int(v) if v != "" else None
            rows.append(row)
    return rows


def transcribe_stem(tg: TrackGroup, engine, pre_processor, start: float = 0.0,
                     duration: Optional[float] = None, model: str = "htdemucs") -> tuple[dict, dict, float]:
    """Separa ``mix.wav`` una vez y transcribe el stem del grupo sin y con PRE-DSP.

    Devuelve ``(raw_t, pre_t, duracion_s)``, donde cada transcripción es el dict de
    ``BasicPitchTranscriber.transcribe`` (sin POST-DSP: el POST se aplica después).
    """
    import librosa

    if not tg.evaluable:
        raise ValueError(f"{tg.track_id}/{tg.group} no es evaluable (faltan MIDI: {tg.missing_midi}).")
    signal, sr = librosa.load(str(tg.mix_path), sr=None, mono=False, offset=start, duration=duration)
    separation = engine.separate(signal, sr=sr)
    if tg.group not in separation.stems:
        raise ValueError(f"{model} no entregó el grupo {tg.group!r}.")
    stem = separation.stems[tg.group].mono(SR_TRANSCRIPTION)

    transcriber = engine.transcriber
    raw_t = transcriber.transcribe(stem.audio, sr=SR_TRANSCRIPTION)
    pre_audio = pre_processor.process(stem.audio.copy(), stem.sr)
    pre_t = transcriber.transcribe(pre_audio, sr=SR_TRANSCRIPTION)
    return raw_t, pre_t, separation.input_audio.duration


def run_conditions(tg: TrackGroup, engine, pre_processor, post_processor,
                   start: float = 0.0, duration: Optional[float] = None,
                   model: str = "htdemucs", midi_dir: Optional[Union[str, Path]] = None) -> list[dict]:
    """Corre las 4 condiciones sobre un track × grupo y devuelve filas del contrato.

    Separa ``mix.wav`` una sola vez y transcribe dos veces (stem y stem con PRE-DSP);
    el POST-DSP se aplica sobre esas mismas notas, así las 4 condiciones comparten audio.
    Si ``midi_dir`` se indica, guarda cada MIDI como ``<track>_<grupo>_<condicion>.mid``
    (los usa el notebook de análisis de errores).
    """
    from .benchmark import evaluate_prediction, load_reference_midis, transpose_midi

    raw_t, pre_t, total = transcribe_stem(tg, engine, pre_processor, start, duration, model)

    def post(t):
        return post_processor.process(t["raw_note_events"], base_midi=t["raw_midi"])[1]

    predictions = {"raw": raw_t["raw_midi"], "pre": pre_t["raw_midi"],
                   "post": post(raw_t), "prepost": post(pre_t)}

    shift = reference_transpose(tg.group)
    reference = load_reference_midis(tg.midi_paths, start=start, duration=total)
    if shift:
        reference = transpose_midi(reference, shift)

    rows = []
    for key, _, pre_on, post_on in CONDITIONS:
        if midi_dir is not None:
            Path(midi_dir).mkdir(parents=True, exist_ok=True)
            predictions[key].write(str(Path(midi_dir) / f"{tg.track_id}_{tg.group}_{key}.mid"))
        m = evaluate_prediction(reference, predictions[key], total)
        rows.append({
            "track_id": tg.track_id, "grupo": tg.group, "stems_ref": "+".join(tg.stem_ids),
            "inst_class": "+".join(sorted(set(tg.inst_classes))), "modelo_demucs": model,
            "condicion": key, "pre_dsp": int(pre_on), "post_dsp": int(post_on),
            "inicio_s": start, "duracion_s": round(total, 3), "ref_transpose": shift,
            **{k: m[k] for k in METRICS}, "n_ref": m["Notas GT"], "n_pred": m["Notas predichas"],
            "estado": m["estado"],
        })
    return rows


# --- Auditoría del POST-DSP (notebook 02) -------------------------------------------

# Basic Pitch descarta por su cuenta las notas más cortas que esto (``minimum_note_length``
# por defecto de ``basic_pitch.inference.predict``, que ``src/`` no cambia).
BASIC_PITCH_MIN_NOTE_S = 0.1277

POST_REASONS = ("ninguno", "duracion", "tesitura", "velocidad")

NOTE_COLUMNS = ("track_id", "grupo", "fuente", "onset_s", "offset_s", "duracion_s",
                "pitch", "velocidad", "motivo")


def post_reason(duration: float, pitch: int, vel_127: int, post) -> str:
    """Qué filtro del ``NotePostProcessor`` elimina esta nota (uno de ``POST_REASONS``).

    ``vel_127`` ya viene escalada a [1, 127], como la compara ``NotePostProcessor.process``.
    """
    if post.min_duration_sec is not None and duration  < post.min_duration_sec:
        return "duracion"
    
    # Filtro 2: Rango de pitch (tesitura)
    if post.pitch_range is not None:
        p_min, p_max = post.pitch_range
        if not (p_min <= pitch <= p_max):
            return "tesitura"
    
    # Filtro 3: Velocidad mínima
    if post.min_velocity is not None:
        thresh_127 = int(round(127 * post.min_velocity)) if post.min_velocity <= 1.0 else int(post.min_velocity)
        if vel_127 < thresh_127:
            return "velocidad"
    
    return "ninguno"
    



def audit_post_dsp(tg: TrackGroup, transcriptions: dict, post_processor) -> list[dict]:
    """Una fila por nota cruda de Basic Pitch con el motivo por el que el POST-DSP la elimina.

    ``transcriptions`` mapea la fuente (``"raw"``, ``"pre"``) al dict de ``transcribe``.
    Las filas siguen ``NOTE_COLUMNS``; se escriben con ``write_rows(..., NOTE_COLUMNS)``.
    """
    rows = []
    for fuente, t in transcriptions.items():
        for onset, offset, pitch, velocity, *_ in t["raw_note_events"]:
            vel_127 = max(1, min(127, int(round(127 * velocity)) if velocity <= 1.0 else int(velocity)))
            rows.append({
                "track_id": tg.track_id, "grupo": tg.group, "fuente": fuente,
                "onset_s": round(onset, 4), "offset_s": round(offset, 4),
                "duracion_s": round(offset - onset, 4), "pitch": int(pitch), "velocidad": vel_127,
                "motivo": post_reason(offset - onset, int(pitch), vel_127, post_processor),
            })
    return rows
