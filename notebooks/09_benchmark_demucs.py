# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.25.1,<0.26",
#     "numpy<2",
#     "scipy",
#     "librosa",
#     "matplotlib",
#     "pyyaml",
#     "pretty_midi",
#     "pedalboard==0.9.21",
#     "basic-pitch",
#     "demucs",
#     "mir_eval",
#     "setuptools<81",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import warnings
    from pathlib import Path

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # resampy (dependencia de Basic Pitch) importa pkg_resources y avisa de su retirada.
    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")

    from src import (
        AudioArtifact,
        DemucsTranscriber,
        available_devices,
        get_audio_pipeline,
        get_note_pipeline,
    )
    from src.benchmark import (
        METRIC_KEYS,
        evaluate_prediction,
        load_reference_midis,
        transpose_midi,
    )

    return (
        AudioArtifact,
        DemucsTranscriber,
        METRIC_KEYS,
        Path,
        available_devices,
        evaluate_prediction,
        get_audio_pipeline,
        get_note_pipeline,
        librosa,
        load_reference_midis,
        mo,
        np,
        plt,
        project_root,
        transpose_midi,
        yaml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 9 · Benchmark por instrumento desde Demucs

    - El audio viene **solo de `mix.wav`**; la carpeta `stems/` del directorio se ignora por completo.
    - `MIDI/<stem_id>.mid` es únicamente la **referencia de evaluación**.
    - Se analiza **un instrumento a la vez**, elegido en la interfaz; su `inst_class` determina el
      grupo Demucs y el preset DSP aplicado.
    - Variantes del diseño 2×2: `Raw stem` (sin DSP), `Pre-DSP only` (audio filtrado, MIDI crudo),
      `Post-DSP only` (audio del stem, notas filtradas), `Pre-DSP + Post-DSP`.
    - Cada etapa del audio (entrada, stem Demucs, audio Pre-DSP) se publica con **reproductor
      escuchable**, forma de onda y espectro para validación directa.
    - Drums queda fuera del benchmark de notas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Carpeta del track

    Selecciona la carpeta con `mix.wav`, `metadata.yaml` y `MIDI/`.
    En modo script (ejecución de punta a punta del `.py`) se usa `data/babyslakh_16k/Track00001`.
    """)
    return


@app.cell
def _(mo, project_root):
    demo_track_dir = project_root / "data/babyslakh_16k/Track00001"
    auto_run = mo.app_meta().mode == "script"

    track_picker = mo.ui.file_browser(
        initial_path=project_root / "data",
        selection_mode="directory",
        multiple=False,
        restrict_navigation=False,
        label="Carpeta del track (mix.wav + metadata.yaml + MIDI/)",
    )
    mo.vstack([track_picker])
    return auto_run, demo_track_dir, track_picker


@app.cell
def _(Path, auto_run, demo_track_dir, mo, track_picker):
    _selected = track_picker.value
    if _selected:
        track_dir = Path(_selected[0].path).expanduser()
    elif auto_run:
        track_dir = demo_track_dir
    else:
        mo.stop(True, mo.md("Selecciona una carpeta de track en el explorador."))

    mix_path = track_dir / "mix.wav"
    metadata_path = track_dir / "metadata.yaml"
    midi_dir = track_dir / "MIDI"

    mo.stop(not track_dir.is_dir(), mo.md(f"No existe la carpeta: `{track_dir}`"))
    mo.stop(not mix_path.is_file(), mo.md(f"Falta `mix.wav` en `{track_dir}`."))
    mo.stop(not metadata_path.is_file(), mo.md(f"Falta `metadata.yaml` en `{track_dir}`."))
    mo.stop(not midi_dir.is_dir(), mo.md(f"Falta la carpeta `MIDI/` en `{track_dir}`."))
    return metadata_path, midi_dir, mix_path, track_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Catálogo de instrumentos y match automático

    Entra al catálogo cada `stems.<id>` no percusivo con `MIDI/<id>.mid` presente.
    El grupo Demucs se infiere así: clase con `bass` → `bass`; programa `Choir`/`Voice`
    o clase `Vocal` → `vocals`; todo lo demás → `other`.
    """)
    return


@app.cell
def _(metadata_path, midi_dir, mo, yaml):
    def _demucs_stem(inst_class: str, program_name: str) -> str:
        if "bass" in inst_class.lower():
            return "bass"
        if (
            "choir" in program_name.lower()
            or "voice" in program_name.lower()
            or "vocal" in inst_class.lower()
        ):
            return "vocals"
        return "other"


    _metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    _stems = (_metadata or {}).get("stems") or {}
    instruments = [
        {
            "stem_id": stem_id,
            "inst_class": info.get("inst_class", "Unknown"),
            "program": info.get("midi_program_name", "Unknown"),
            "midi_path": midi_dir / f"{stem_id}.mid",
            "demucs_stem": _demucs_stem(
                info.get("inst_class", ""), info.get("midi_program_name", "")
            ),
        }
        for stem_id, info in sorted(_stems.items())
        if not info.get("is_drum") and (midi_dir / f"{stem_id}.mid").is_file()
    ]
    mo.stop(not instruments, mo.md("El track no tiene instrumentos tonales con MIDI en `MIDI/`."))

    mo.ui.table(
        [
            {
                "Stem": item["stem_id"],
                "Instrumento": item["inst_class"],
                "Programa MIDI": item["program"],
                "Archivo MIDI": item["midi_path"].name,
                "Stem Demucs": item["demucs_stem"],
            }
            for item in instruments
        ],
        selection=None,
    )
    return (instruments,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Selección y ventana

    Se elige un solo instrumento: la fila del `metadata.yaml` que se evalúa. Su `inst_class`
    determina el grupo Demucs (bass / vocals / other) y el preset DSP.
    La ventana se recorta de `mix.wav` antes de la separación, así que todas las variantes
    comparten exactamente el mismo audio de partida en tiempo.
    """)
    return


@app.cell
def _(available_devices, instruments, mo):
    _instrument_options = {
        f"{item['inst_class']} ({item['stem_id']})": item["stem_id"] for item in instruments
    }
    instrument_selector = mo.ui.dropdown(
        options=_instrument_options,
        value=next(iter(_instrument_options)),
        label="Instrumento a analizar",
    )
    offset_control = mo.ui.number(start=0, value=0, label="Inicio (segundos)")
    duration_control = mo.ui.number(
        start=1, stop=300, value=30, label="Duración del fragmento (segundos)"
    )
    device_selector = mo.ui.dropdown(
        options=available_devices(), value="cpu", label="Dispositivo Demucs"
    )
    separate_button = mo.ui.run_button(label="Separar con HT-Demucs")

    mo.vstack(
        [
            instrument_selector,
            mo.hstack([offset_control, duration_control, device_selector], wrap=True),
            separate_button,
        ]
    )
    return (
        device_selector,
        duration_control,
        instrument_selector,
        offset_control,
        separate_button,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Presets DSP por instrumento

    Los valores provienen de `src/transcriber.py` (`get_audio_pipeline`, `get_note_pipeline`)
    porque no existe un notebook 09 con parámetros por instrumento. Si un futuro 09 produce
    otros valores, se sustituyen **solo** en esta celda; el resto del flujo no cambia.
    """)
    return


@app.cell
def _(
    get_audio_pipeline,
    get_note_pipeline,
    instrument_selector,
    instruments,
    mo,
):
    instrument = next(item for item in instruments if item["stem_id"] == instrument_selector.value)
    pre_processor = get_audio_pipeline(instrument["inst_class"])
    post_processor = get_note_pipeline(instrument["inst_class"])


    def _pre_summary(pre) -> dict:
        summary = {"hp": None, "lp": None, "gate": None}
        for plugin in pre.board or []:
            name = type(plugin).__name__
            if "Highpass" in name:
                summary["hp"] = getattr(plugin, "cutoff_frequency_hz", None)
            elif "Lowpass" in name:
                summary["lp"] = getattr(plugin, "cutoff_frequency_hz", None)
            elif "Gate" in name:
                summary["gate"] = getattr(plugin, "threshold_db", None)
        return summary


    _summary = _pre_summary(pre_processor)
    mo.ui.table(
        [
            {
                "Instrumento": f"{instrument['inst_class']} ({instrument['stem_id']})",
                "Stem Demucs": instrument["demucs_stem"],
                "Pre-DSP": pre_processor.name,
                "HP (Hz)": _summary["hp"],
                "LP (Hz)": _summary["lp"],
                "Gate (dB)": _summary["gate"],
                "Tesitura": post_processor.pitch_range
                if post_processor.pitch_range is not None
                else "—",
                "Duración mín. (s)": post_processor.min_duration_sec,
                "Velocidad mín.": post_processor.min_velocity
                if post_processor.min_velocity is not None
                else "—",
            }
        ],
        selection=None,
    )
    return instrument, post_processor, pre_processor


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Separación con HT-Demucs

    Se carga la ventana de `mix.wav` y se separa en los cuatro grupos del modelo
    (`drums`, `vocals`, `bass`, `other`). El botón evita repetir la separación al mover otros controles.
    """)
    return


@app.cell
def _(
    AudioArtifact,
    DemucsTranscriber,
    auto_run,
    device_selector,
    duration_control,
    librosa,
    mix_path,
    mo,
    offset_control,
    separate_button,
):
    mo.stop(
        not (auto_run or separate_button.value),
        mo.md("Pulsa **Separar con HT-Demucs** para ejecutar la separación."),
    )

    _signal, _sr = librosa.load(
        str(mix_path), sr=None, mono=False, offset=offset_control.value,
        duration=duration_control.value,
    )
    mo.stop(_signal.size == 0, mo.md("El fragmento seleccionado está vacío."))
    input_artifact = AudioArtifact(_signal, _sr)

    with mo.status.spinner(title="Separando el fragmento con HT-Demucs…"):
        engine = DemucsTranscriber(device=device_selector.value)
        separation = engine.separate(input_artifact.audio, sr=input_artifact.sr)

    mo.vstack(
        [
            mo.ui.table(
                [
                    {
                        "Etapa": "Entrada",
                        "sr": input_artifact.sr,
                        "Duración (s)": round(input_artifact.duration, 3),
                    },
                    {
                        "Etapa": "Demucs",
                        "sr": separation.input_audio.sr,
                        "Duración (s)": round(separation.input_audio.duration, 3),
                    },
                ],
                selection=None,
            ),
            mo.md(
                "El audio analizado proviene de `mix.wav`; este notebook no abre ningún archivo "
                "de `stems/`."
            ),
        ]
    )
    return engine, input_artifact, separation


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · Etapas del audio: escuchar, forma de onda y espectro

    Cada etapa del instrumento elegido se publica con su reproductor y una figura de dos paneles:
    forma de onda y espectrograma STFT en dB. Así se valida de oído y de vista qué aporta cada paso
    —la separación Demucs y el filtrado Pre-DSP— antes de mirar métricas.

    Los reproductores normalizan la ganancia para escucha cómoda; la forma de onda y el espectro
    muestran los niveles reales de cada etapa.
    """)
    return


@app.cell
def _(
    AudioArtifact,
    input_artifact,
    instrument,
    librosa,
    mo,
    np,
    plt,
    pre_processor,
    separation,
):
    mo.stop(
        instrument["demucs_stem"] not in separation.stems,
        mo.md(f"El modelo no produjo el stem `{instrument['demucs_stem']}`."),
    )

    stem_artifact = separation.stems[instrument["demucs_stem"]].mono(22050)
    pre_audio = pre_processor.process(stem_artifact.audio.copy(), stem_artifact.sr)

    stages = [
        ("Entrada · mix.wav", input_artifact.mono(22050)),
        (f"Stem Demucs · {instrument['demucs_stem']}", stem_artifact),
        ("Pre-DSP · audio filtrado", AudioArtifact(pre_audio, stem_artifact.sr)),
    ]


    def _spectrogram_db(audio, sr, hop=512):
        spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop))
        return librosa.amplitude_to_db(spectrum, ref=np.max), sr / 2


    _stage_blocks = []
    for _title, _artifact in stages:
        _db, _f_max = _spectrogram_db(_artifact.audio, _artifact.sr)
        _fig, _axes = plt.subplots(
            2, 1, figsize=(11, 4.4), gridspec_kw={"height_ratios": [1, 2]}
        )
        _axes[0].plot(
            np.arange(_artifact.audio.size) / _artifact.sr,
            _artifact.audio,
            linewidth=0.6,
            color="C0",
        )
        _axes[0].set_ylabel("amplitud")
        _axes[0].set_title(
            f"{_title} · {_artifact.sr} Hz · {_artifact.duration:.2f} s · "
            f"pico {np.abs(_artifact.audio).max():.3f}"
        )
        _axes[0].grid(alpha=0.3)
        _axes[1].imshow(
            _db,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[0, _artifact.duration, 0, _f_max],
        )
        _axes[1].set_xlabel("tiempo (s)")
        _axes[1].set_ylabel("frecuencia (Hz)")
        _fig.tight_layout()
        _stage_blocks.extend(
            [
                mo.md(f"**{_title}**"),
                mo.audio(_artifact.audio, rate=_artifact.sr),
                _fig,
            ]
        )
        plt.close(_fig)

    mo.vstack(_stage_blocks)
    return pre_audio, stem_artifact


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7 · Benchmark 2×2 del instrumento elegido

    - `Raw stem`: audio del grupo Demucs, MIDI crudo.
    - `Pre-DSP only`: audio filtrado, MIDI crudo.
    - `Post-DSP only`: audio del grupo, notas filtradas.
    - `Pre-DSP + Post-DSP`: audio filtrado y notas filtradas.

    La referencia es el `MIDI/<stem_id>.mid` del instrumento, recortado a la misma ventana.
    Para el bajo se transpone **−12 semitonos**: el render de BabySlakh suena una octava por
    debajo de su MIDI (11.84–11.99 semitonos medidos en 9 de 10 tracks). Sin esa corrección,
    `F_no` es 0 por desajuste de octava aunque la transcripción sea correcta.
    """)
    return


@app.cell
def _(
    engine,
    evaluate_prediction,
    instrument,
    load_reference_midis,
    mo,
    offset_control,
    post_processor,
    pre_audio,
    separation,
    stem_artifact,
    transpose_midi,
):
    VARIANT_ORDER = ("Raw stem", "Pre-DSP only", "Post-DSP only", "Pre-DSP + Post-DSP")
    total_duration = separation.input_audio.duration

    raw_transcription = engine.transcriber.transcribe(stem_artifact.audio, sr=22050)
    pre_transcription = engine.transcriber.transcribe(pre_audio, sr=22050)

    predictions = {
        "Raw stem": raw_transcription["raw_midi"],
        "Pre-DSP only": pre_transcription["raw_midi"],
        "Post-DSP only": post_processor.process(
            raw_transcription["raw_note_events"], base_midi=raw_transcription["raw_midi"]
        )[1],
        "Pre-DSP + Post-DSP": post_processor.process(
            pre_transcription["raw_note_events"], base_midi=pre_transcription["raw_midi"]
        )[1],
    }
    reference = load_reference_midis(
        [instrument["midi_path"]], start=offset_control.value, duration=total_duration
    )
    # El render de BabySlakh suena una octava por debajo del MIDI en el bajo
    # (verificado en 9 de 10 tracks): calibrar la referencia, no el modelo.
    reference_transpose = -12 if "bass" in instrument["inst_class"].lower() else 0
    if reference_transpose:
        reference = transpose_midi(reference, reference_transpose)

    rows = []
    _raw_f_no = None
    with mo.status.spinner(title="Evaluando variantes…"):
        for _variant in VARIANT_ORDER:
            _metrics = evaluate_prediction(reference, predictions[_variant], total_duration)
            if _variant == "Raw stem":
                _raw_f_no = _metrics["F_no"]
            _delta = None
            if _metrics["F_no"] is not None and _raw_f_no is not None:
                _delta = _metrics["F_no"] - _raw_f_no
            rows.append(
                {
                    "Instrumento": f"{instrument['inst_class']} ({instrument['stem_id']})",
                    "Stem Demucs": instrument["demucs_stem"],
                    "Variante": _variant,
                    "P_no": _metrics["P_no"],
                    "R_no": _metrics["R_no"],
                    "F_no": _metrics["F_no"],
                    "P": _metrics["P"],
                    "R": _metrics["R"],
                    "F": _metrics["F"],
                    "Acc": _metrics["Acc"],
                    "Notas GT": _metrics["Notas GT"],
                    "Notas predichas": _metrics["Notas predichas"],
                    "ΔF_no": _delta,
                    "estado": _metrics["estado"],
                }
            )
    return VARIANT_ORDER, reference_transpose, rows, total_duration


@app.cell
def _(
    METRIC_KEYS,
    VARIANT_ORDER,
    instrument,
    mo,
    offset_control,
    reference_transpose,
    rows,
    total_duration,
    track_dir,
):
    _display_rows = []
    for _row in rows:
        _clean = dict(_row)
        for _key in METRIC_KEYS:
            if _clean.get(_key) is not None:
                _clean[_key] = round(_clean[_key], 4)
        if _clean["ΔF_no"] is not None:
            _clean["ΔF_no"] = round(_clean["ΔF_no"], 4)
        _display_rows.append(_clean)

    mo.vstack(
        [
            mo.md(
                f"**Stem:** `{track_dir.name}` · fragmento "
                f"{offset_control.value:.2f}–{offset_control.value + total_duration:.2f} s "
                f"({total_duration:.2f} s) · {instrument['inst_class']} "
                f"({instrument['stem_id']}) · {len(VARIANT_ORDER)} variantes contra su propio "
                f"`MIDI/<stem_id>.mid`."
                + (f" Referencia transpuesta {reference_transpose:+d} semitonos."
                   if reference_transpose else "")
            ),
            mo.ui.table(_display_rows, selection=None, show_download=True),
        ]
    )
    return


@app.cell
def _(VARIANT_ORDER, instrument, np, plt, rows):
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _values = [row["F_no"] if row["F_no"] is not None else 0.0 for row in rows]
    _bars = _ax.barh(
        np.arange(len(VARIANT_ORDER)),
        _values,
        color=[f"C{_index}" for _index in range(len(VARIANT_ORDER))],
    )
    _ax.bar_label(_bars, fmt="%.3f", padding=3, fontsize=8)
    _ax.set_yticks(np.arange(len(VARIANT_ORDER)))
    _ax.set_yticklabels([row["Variante"] for row in rows])
    _ax.invert_yaxis()
    _ax.set_xlim(0, 1)
    _ax.set_xlabel("F_no (inicio + altura)")
    _ax.set_title(
        f"F_no por variante · {instrument['inst_class']} ({instrument['stem_id']})"
    )
    _ax.grid(axis="x", alpha=0.3)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8 · Lectura y limitaciones

    - El análisis es de **un instrumento por ejecución**: al cambiarlo se recalcula la separación y
      las cuatro variantes de esa fila del `metadata.yaml`.
    - `other` agrupa varios instrumentos en un solo grupo Demucs, así que su predicción no está
      aislada: su F_no también mide cuánto material ajeno descarta el pipeline.
    - La referencia es el MIDI del instrumento en la carpeta, no los stems del directorio.
    - La calibración de octava es una propiedad del **render del dataset**, no de la métrica:
      `Bass → −12` se aplica una vez por clase y se documenta, nunca se ajusta por track
      maximizando `F_no`.
    - `drums` no se evalúa: las métricas de notas no aplican a percusión.
    - Tolerancias heredadas del notebook 06: 50 ms de onset, 50 cents de altura y frames de 10 ms.
    - `ΔF_no` es la diferencia contra `Raw stem` del mismo instrumento; negativo significa que la
      variante pierde notas correctas.
    - Los reproductores normalizan ganancia; la forma de onda y el espectro muestran los niveles reales.
    """)
    return


if __name__ == "__main__":
    app.run()
