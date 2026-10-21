import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    from datetime import datetime

    import marimo as mo
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    from pedalboard import HighpassFilter, LowpassFilter, NoiseGate

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src import AudioPreProcessor, DemucsTranscriber, DSPVariant, NotePostProcessor, available_devices
    from src.benchmark import benchmark_variants, load_reference_midis
    from src.spectral import HOP_LENGTH, SAMPLE_RATE, spectral_comparison

    return (
        AudioPreProcessor, DemucsTranscriber, DSPVariant, HighpassFilter,
        LowpassFilter, NoiseGate, NotePostProcessor, Path, Rectangle,
        HOP_LENGTH, SAMPLE_RATE, available_devices, benchmark_variants, datetime, librosa,
        load_reference_midis, mo, np, plt, project_root, spectral_comparison,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 07 · HT-Demucs + DSP + Basic Pitch

    Separa una mezcla una vez, escucha sus stems y compara variantes DSP y MIDI.
    Empieza con **10 segundos**: la primera separación puede descargar los pesos de
    HT-Demucs. Los modelos se ejecutan con botones; los controles DSP actualizan
    audio y gráficos. Cambiar el DSP invalida la transcripción anterior.

    **Frecuencias:** entrada nativa → estéreo a 44.1 kHz → HT-Demucs → copia mono
    a 22.05 kHz → etapas DSP → Basic Pitch → filtrado de notas.
    Remuestrear 16 kHz no recupera frecuencias ausentes. `other` agrupa instrumentos;
    `drums` se ofrece para DSP y escucha, sin transcripción en esta demostración.
    """)
    return


@app.cell
def _(available_devices, mo, project_root):
    source_form = mo.ui.dictionary({
        "audio": mo.ui.file_browser(
            initial_path=project_root / "data" if (project_root / "data").is_dir() else project_root,
            filetypes=[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"],
            selection_mode="file",
            multiple=False,
            restrict_navigation=False,
            label="Seleccionar audio"),
        "offset": mo.ui.number(start=0, value=0, label="Inicio (segundos)"),
        "duration": mo.ui.number(start=1, stop=120, value=10,
                                 label="Duración del fragmento (segundos)"),
        "device": mo.ui.dropdown(options=available_devices(), value="cpu",
                                 label="Dispositivo Demucs (mps: GPU Apple · cuda: NVIDIA)"),
    }).form(submit_button_label="Separar con HT-Demucs")
    source_form
    return (source_form,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Dispositivo:** CPU utiliza el procesador nativo (ARM en el M1).
    Si está disponible, `mps` permite ejecutar Demucs con la GPU de Apple Silicon;
    `cuda` corresponde a NVIDIA. El selector muestra solo las opciones detectadas.
    Basic Pitch mantiene su backend independiente; esta selección afecta a Demucs.
    """)
    return


@app.cell
def _(DemucsTranscriber, Path, librosa, mo, source_form):
    mo.stop(source_form.value is None, mo.md("Selecciona un audio y pulsa **Separar con HT-Demucs**."))
    source_config = dict(source_form.value)
    selected_files = source_config.pop("audio")
    mo.stop(not selected_files, mo.md("Selecciona un archivo de audio en el explorador."))
    source_path = Path(selected_files[0].path).expanduser()
    source_config["path"] = str(source_path)
    mo.stop(not source_path.is_file(), mo.md(f"No existe el archivo: `{source_path}`"))
    with mo.status.spinner(title="Separando el fragmento con HT-Demucs…"):
        input_signal, input_sr = librosa.load(
            str(source_path), sr=None, mono=False,
            offset=source_config["offset"], duration=source_config["duration"],
        )
        mo.stop(input_signal.size == 0, mo.md("El fragmento seleccionado está vacío."))
        engine = DemucsTranscriber(device=source_config["device"])
        separation = engine.separate(input_signal, sr=input_sr)
    return engine, separation, source_config, source_path


@app.cell
def _(mo, separation):
    mo.ui.table([
        {"etapa": "Entrada", "Hz": separation.input_audio.sr,
         "segundos": separation.input_audio.duration},
        {"etapa": "Demucs (estéreo)", "Hz": separation.demucs_input.sr,
         "segundos": separation.demucs_input.duration},
        {"etapa": "DSP / Basic Pitch (mono)", "Hz": 22050,
         "segundos": separation.demucs_input.duration},
    ], selection=None)
    return


@app.cell
def _(mo, separation):
    stem_selector = mo.ui.dropdown(options=list(separation.stems), value="bass", label="Stem")
    stem_selector
    return (stem_selector,)


@app.cell
def _(audio_player, mo, separation, stem_selector):
    _stem_audio = separation.stems[stem_selector.value]
    mo.vstack([
        mo.md(f"### {stem_selector.value} · salida estéreo de Demucs · {_stem_audio.sr} Hz"),
        audio_player(_stem_audio),
    ])
    return


@app.cell
def _(mo):
    hp_control = mo.ui.slider(20, 1000, step=5, value=35, show_value=True, label="Paso alto (Hz)")
    lp_control = mo.ui.slider(1100, 10000, step=100, value=5000, show_value=True, label="Paso bajo (Hz)")
    gate_control = mo.ui.slider(-80, -10, step=1, value=-42, show_value=True, label="Noise gate (dB)")
    duration_control = mo.ui.slider(0.01, 0.3, step=0.01, value=0.05,
                                    show_value=True, label="Duración mínima de nota (s)")
    variant_selector = mo.ui.multiselect(
        options=["raw", "filtered", "alternative"],
        value=["raw", "filtered", "alternative"], label="Variantes a comparar")
    mo.vstack([
        mo.md("""### Variantes independientes
        - **raw:** stem mono sin DSP.
        - **filtered:** paso alto → paso bajo → noise gate → Basic Pitch → filtro de duración.
        - **alternative:** paso alto → paso bajo → Basic Pitch → filtro de duración.

        Cada rama comienza en el mismo stem; cada efecto es una etapa recuperable.
        Los ajustes son puntos de partida experimentales, no presets óptimos por instrumento.
        """),
        mo.hstack([hp_control, lp_control, gate_control, duration_control], wrap=True),
        variant_selector,
    ])
    return duration_control, gate_control, hp_control, lp_control, variant_selector


@app.cell
def _(
    AudioPreProcessor, DSPVariant, HighpassFilter, LowpassFilter, NoiseGate,
    NotePostProcessor, duration_control, engine, gate_control, hp_control,
    lp_control, mo, separation, stem_selector, variant_selector,
):
    mo.stop(not variant_selector.value, mo.md("Selecciona al menos una variante."))

    def make_stages(with_gate):
        # Instancias nuevas por rama: Pedalboard puede conservar estado interno.
        stages = [
            AudioPreProcessor(name="highpass", board=[HighpassFilter(cutoff_frequency_hz=hp_control.value)]),
            AudioPreProcessor(name="lowpass", board=[LowpassFilter(cutoff_frequency_hz=lp_control.value)]),
        ]
        if with_gate:
            stages.append(AudioPreProcessor(name="gate", board=[
                NoiseGate(threshold_db=gate_control.value, ratio=6.0)]))
        return stages

    variant_configs = {
        "raw": DSPVariant("raw"),
        "filtered": DSPVariant("filtered", make_stages(True),
            NotePostProcessor(min_duration_sec=duration_control.value)),
        "alternative": DSPVariant("alternative", make_stages(False),
            NotePostProcessor(min_duration_sec=duration_control.value)),
    }
    dsp_results = engine.process_variants(separation, {
        stem_selector.value: [variant_configs[name] for name in variant_selector.value],
    })
    return (dsp_results,)


@app.cell
def _(dsp_results, mo, stem_selector):
    stage_variant = mo.ui.dropdown(
        options=list(dsp_results[stem_selector.value]),
        value=next(iter(dsp_results[stem_selector.value])), label="Rama para inspeccionar etapas")
    stage_variant
    return (stage_variant,)


@app.cell
def _(mo):
    spectral_selector = mo.ui.dropdown(
        options=["STFT", "Mel", "log-Mel"], value="log-Mel",
        label="Representación espectral")
    mo.vstack([
        mo.md("""### Representación espectral
        Mel agrupa frecuencias; log-Mel muestra su potencia en dB y facilita
        inspeccionar cambios de energía, aunque puede ocultar detalles finos.
        STFT conserva la vista con eje de frecuencia logarítmico.
        Cambiar la vista actualiza solo los gráficos: no modifica el audio
        enviado a Basic Pitch ni mejora por sí mismo la transcripción.
        Todas las vistas usan 22.05 kHz, FFT de 2048 muestras y salto de 256;
        Mel utiliza 128 bandas entre 0 y 11025 Hz. Las escalas se comparten
        y la referencia es siempre el stem sin DSP.
        """), spectral_selector,
    ])
    return (spectral_selector,)


@app.cell
def _(HOP_LENGTH, Rectangle, SAMPLE_RATE, librosa, mo, np, plt, spectral_comparison):
    def audio_player(artifact):
        import io
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, artifact.audio.T, artifact.sr, format="WAV", subtype="FLOAT")
        buffer.seek(0)
        return mo.audio(buffer)

    def compare_audio(artifacts, reference_audio, representation):
        mono = [artifact.mono() for artifact in artifacts.values()]
        comparison = spectral_comparison(artifacts, reference_audio, representation)
        amplitude = max(1e-4, max(float(np.max(np.abs(a.audio))) for a in mono))
        fig, axes = plt.subplots(len(mono), 2, figsize=(14, 3 * len(mono)),
                                 squeeze=False, sharex=True, sharey="col", layout="constrained")
        for index, (name, artifact) in enumerate(zip(artifacts, mono)):
            # Diezmado solo para dibujar: las inferencias usan todas las muestras.
            step = max(1, len(artifact.audio) // 20000)
            time = np.arange(0, len(artifact.audio), step) / artifact.sr
            axes[index, 0].plot(time, artifact.audio[::step], linewidth=0.5)
            axes[index, 0].set(title=f"{name} · forma de onda", ylim=(-amplitude, amplitude),
                               ylabel="Amplitud", xlabel="Tiempo (s)")
            image = librosa.display.specshow(
                comparison.matrices[name], sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                x_axis="time", y_axis=comparison.y_axis, fmin=0, fmax=SAMPLE_RATE / 2,
                cmap="magma", vmin=comparison.vmin, vmax=comparison.vmax, ax=axes[index, 1])
            axes[index, 1].set_title(f"{name} · {representation}")
            axes[index, 0].set_xlim(0, reference_audio.duration)
        fig.colorbar(image, ax=list(axes[:, 1]), label=comparison.label)
        return fig

    def compare_midi(branches):
        fig, axes = plt.subplots(len(branches), 2, figsize=(14, 3 * len(branches)),
                                 squeeze=False, sharex=True, sharey=True, layout="constrained")
        for row, (name, result) in enumerate(branches.items()):
            for col, key in enumerate(("raw_note_events", "note_events")):
                events = result.transcription[key]
                axis = axes[row, col]
                for onset, offset, pitch, *_ in events:
                    axis.add_patch(Rectangle((onset, pitch - 0.4), offset - onset, 0.8,
                                             facecolor=f"C{row % 10}", alpha=0.7))
                axis.set(title=f"{name} · {'crudo' if col == 0 else 'filtrado'} · {len(events)} notas",
                         xlim=(0, result.final_audio.duration), ylim=(20, 109),
                         xlabel="Tiempo relativo al fragmento (s)", ylabel="Nota MIDI")
                axis.grid(alpha=0.2)
        return fig

    return audio_player, compare_audio, compare_midi


@app.cell
def _(compare_audio, dsp_results, mo, plt, separation, spectral_selector, stage_variant, stem_selector):
    stage_artifacts = dsp_results[stem_selector.value][stage_variant.value].stages
    _fig = compare_audio(stage_artifacts, separation.stems[stem_selector.value], spectral_selector.value)
    mo.output.replace(_fig)
    plt.close(_fig)
    return (stage_artifacts,)


@app.cell
def _(audio_player, mo, stage_artifacts):
    mo.vstack([
        mo.hstack([mo.md(f"**{_name}** · {_artifact.sr} Hz"),
                   audio_player(_artifact)], widths=[1, 3], wrap=True)
        for _name, _artifact in stage_artifacts.items()
    ])
    return


@app.cell
def _(compare_audio, dsp_results, mo, plt, separation, spectral_selector, stem_selector):
    _artifacts = {_name: _result.final_audio
                  for _name, _result in dsp_results[stem_selector.value].items()}
    _fig = compare_audio(_artifacts, separation.stems[stem_selector.value], spectral_selector.value)
    mo.output.replace(mo.vstack([mo.md("### Comparación de las salidas de las ramas"), _fig]))
    plt.close(_fig)
    return


@app.cell
def _(mo):
    transcribe_button = mo.ui.run_button(label="Transcribir variantes con Basic Pitch")
    mo.vstack([
        mo.md("""### Comparación MIDI
        Pulsa el botón después de ajustar el DSP. Los gráficos muestran notas detectadas,
        no una medida de exactitud: menos notas no implica mejor transcripción.
        El filtrado de notas utiliza el `NotePostProcessor` existente; su reconstrucción
        actual de MIDI no conserva pitch bends ni todos los metadatos del MIDI crudo.
        """), transcribe_button,
    ])
    return (transcribe_button,)


@app.cell
def _(dsp_results, engine, mo, stem_selector, transcribe_button):
    mo.stop(stem_selector.value == "drums", mo.md("Drums: utiliza DSP, escucha y exportación de audio."))
    mo.stop(not transcribe_button.value, mo.md("Pulsa **Transcribir variantes** para generar MIDI con los ajustes actuales."))
    with mo.status.spinner(title="Transcribiendo las variantes…"):
        transcription_results = engine.transcribe_variants(dsp_results)
    return (transcription_results,)


@app.cell
def _(compare_midi, mo, plt, stem_selector, transcription_results):
    _branches = transcription_results[stem_selector.value]
    _fig = compare_midi(_branches)
    mo.output.replace(mo.vstack([
        _fig,
        mo.ui.table([
            {"variante": _name, "notas crudas": len(_result.transcription["raw_note_events"]),
             "notas filtradas": len(_result.transcription["note_events"])}
            for _name, _result in _branches.items()
        ], selection=None),
    ]))
    plt.close(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Benchmark de transcripción · métricas del 06

    Selecciona los MIDI que corresponden al **stem actual**. Para `other`, reúne
    los instrumentos presentes en esa agrupación; no uses el MIDI de la mezcla
    completa para evaluar únicamente el bajo. Las pistas percusivas se excluyen.
    En BabySlakh los MIDI originales ya comparten el origen temporal del audio.
    En otros datos, las referencias también deben estar alineadas con el audio.

    **F_no** exige altura e inicio de nota; **F** exige además su final;
    **Acc** compara alturas activas por frames. No son una métrica llamada F0.
    Tolerancias: inicio 50 ms, altura 50 cents, final máximo entre 20 % de duración
    y 50 ms; frames de 10 ms. Las notas sostenidas se recortan en los bordes.
    Los deltas comparan cada salida contra el MIDI del modelo de `raw`.
    La evaluación reutiliza los MIDI: no vuelve a ejecutar los modelos.
    """)
    return


@app.cell
def _(mo, source_path):
    import yaml

    _metadata_path = source_path.parent / "metadata.yaml"
    _midi_dir = source_path.parent / "MIDI"
    _metadata_message = None
    _metadata = {}
    if _metadata_path.is_file():
        try:
            _loaded = yaml.safe_load(_metadata_path.read_text(encoding="utf-8"))
            if isinstance(_loaded, dict):
                _metadata = _loaded.get("stems", {}) or {}
        except (OSError, yaml.YAMLError) as _error:
            _metadata_message = mo.md(f"No se pudieron leer los metadatos: `{_error}`")
    _records = []
    if _midi_dir.is_dir():
        for _path in sorted(_midi_dir.iterdir()):
            if _path.suffix.lower() in (".mid", ".midi"):
                _info = _metadata.get(_path.stem, {})
                _records.append({
                    "Archivo MIDI": _path.name,
                    "Clase": _info.get("inst_class", "Sin metadatos"),
                    "Instrumento": _info.get("midi_program_name", "Sin metadatos"),
                    "Percusión": _info.get("is_drum", False),
                })
    mo.vstack([
        mo.md("**Referencias disponibles en el track**"),
        *([_metadata_message] if _metadata_message is not None else []),
        mo.ui.table(_records, selection=None) if _records
        else mo.md("Puedes seleccionar MIDI alineados desde otra carpeta."),
    ])
    return


@app.cell
def _(mo, source_path, stem_selector):
    mo.stop(stem_selector.value == "drums", mo.md("Drums queda fuera del benchmark de notas."))
    _initial = source_path.parent / "MIDI"
    reference_selector = mo.ui.file_browser(
        initial_path=_initial if _initial.is_dir() else source_path.parent,
        filetypes=[".mid", ".midi"], selection_mode="file", multiple=True,
        restrict_navigation=False,
        label=f"MIDI de referencia para {stem_selector.value} (selección explícita)",
    )
    reference_selector
    return (reference_selector,)


@app.cell
def _(mo):
    benchmark_button = mo.ui.run_button(label="Evaluar benchmark")
    benchmark_button
    return (benchmark_button,)


@app.cell
def _(
    benchmark_button, benchmark_variants, load_reference_midis, mo,
    reference_selector, separation, source_config, stem_selector, transcription_results,
):
    mo.stop(not reference_selector.value, mo.md("Selecciona uno o varios MIDI de referencia para evaluar."))
    mo.stop(not benchmark_button.value, mo.md("Pulsa **Evaluar benchmark** con la referencia seleccionada."))
    reference_paths = [_entry.path for _entry in reference_selector.value]
    benchmark_duration = separation.input_audio.duration
    reference_midi = load_reference_midis(
        reference_paths, start=source_config["offset"], duration=benchmark_duration)
    benchmark_rows = benchmark_variants(
        transcription_results[stem_selector.value], reference_midi, benchmark_duration)
    _messages = [mo.md(
        f"**Stem:** {stem_selector.value} · **Fragmento:** "
        f"{source_config['offset']:.3f}–{source_config['offset'] + benchmark_duration:.3f} s · "
        f"**Referencias:** {', '.join(_path.name for _path in reference_paths)}")]
    if benchmark_rows and benchmark_rows[0]["estado"] == "sin referencia evaluable":
        _messages.append(mo.md("**sin referencia evaluable**: no hay notas no percusivas en este fragmento. Las métricas quedan sin calcular."))
    if "raw" not in transcription_results[stem_selector.value]:
        _messages.append(mo.md("Sin variante `raw`: se muestran métricas absolutas y los deltas quedan sin calcular."))
    _messages.append(mo.ui.table(benchmark_rows, selection=None, show_download=True))
    mo.vstack(_messages)
    return benchmark_duration, benchmark_rows, reference_midi, reference_paths


@app.cell
def _(mo):
    export_audio_button = mo.ui.run_button(label="Exportar separación y etapas WAV")
    export_midi_button = mo.ui.run_button(label="Exportar WAV y MIDI de la comparación")
    mo.vstack([
        mo.md("### Exportación opcional\nCada exportación crea una carpeta nueva en `outputs/07_demucs/`. Los tiempos MIDI son relativos al fragmento."),
        mo.hstack([export_audio_button, export_midi_button]),
    ])
    return export_audio_button, export_midi_button


@app.cell
def _(
    datetime, duration_control, engine, gate_control, hp_control, lp_control,
    project_root, source_config, source_path, stem_selector, variant_selector,
):
    def export_experiment(separated, results, kind):
        import json

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder = project_root / "outputs/07_demucs" / f"{source_path.stem}_{kind}_{stamp}"
        engine.export(separated, results, folder)
        (folder / "source.json").write_text(json.dumps(source_config, indent=2), encoding="utf-8")
        settings = {
            "stem": stem_selector.value, "variants": list(variant_selector.value),
            "highpass_hz": hp_control.value, "lowpass_hz": lp_control.value,
            "gate_db": gate_control.value, "gate_ratio": 6.0,
            "min_note_duration_seconds": duration_control.value,
        }
        (folder / "dsp_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        return folder

    return (export_experiment,)


@app.cell
def _(dsp_results, export_audio_button, export_experiment, mo, separation):
    mo.stop(not export_audio_button.value)
    _folder = export_experiment(separation, dsp_results, "audio")
    mo.md(f"Audio exportado a `{_folder}`")
    return


@app.cell
def _(export_experiment, export_midi_button, mo, separation, transcription_results):
    mo.stop(not export_midi_button.value)
    _folder = export_experiment(separation, transcription_results, "midi")
    mo.md(f"Audio y MIDI exportados a `{_folder}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Uso desde Python

    ```python
    from src import DemucsTranscriber, DSPVariant, AudioPreProcessor
    from pedalboard import HighpassFilter

    engine = DemucsTranscriber()
    separation = engine.separate("mix.wav")
    results = engine.process_variants(separation, {
        "bass": [DSPVariant("raw"), DSPVariant("filtered", [
            AudioPreProcessor(name="highpass", board=[HighpassFilter(cutoff_frequency_hz=35)])
        ])],
        "vocals": [DSPVariant("raw")],
    })
    first_dsp = results["bass"]["filtered"].stages["highpass"]
    transcribed = engine.transcribe_variants(results)
    engine.export(separation, transcribed, "outputs/my_experiment")
    ```

    Cada etapa tiene `.audio` y `.sr`; la salida de Demucs está en
    `separation.stems["bass"]`. Puedes llamar nuevamente a `process_variants`
    sobre la misma separación. Para añadir DSP personalizado usa funciones
    `(audio, sr) -> audio` que conserven forma, duración y frecuencia.
    """)
    return


if __name__ == "__main__":
    app.run()
