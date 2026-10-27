# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.25.1,<0.26",
#     "numpy<2",
#     "scipy",
#     "librosa",
#     "soundfile",
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
    import io
    import sys
    import warnings
    from dataclasses import replace
    from pathlib import Path

    import librosa
    import librosa.display
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    from pedalboard import HighpassFilter, LowpassFilter, NoiseGate

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # resampy (dependencia de Basic Pitch) importa pkg_resources y avisa de su retirada.
    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")

    from src import (
        AudioArtifact,
        AudioPreProcessor,
        DSPVariant,
        DemucsTranscriber,
        NotePostProcessor,
        available_devices,
    )
    from src.benchmark import benchmark_variants, load_reference_midis
    from src.spectral import HOP_LENGTH, SAMPLE_RATE, spectral_comparison


    return (
        AudioArtifact,
        AudioPreProcessor,
        DSPVariant,
        DemucsTranscriber,
        HOP_LENGTH,
        HighpassFilter,
        LowpassFilter,
        NoiseGate,
        NotePostProcessor,
        Path,
        SAMPLE_RATE,
        available_devices,
        benchmark_variants,
        io,
        librosa,
        load_reference_midis,
        mo,
        np,
        plt,
        project_root,
        replace,
        sf,
        spectral_comparison,
    )


@app.cell
def _(
    HOP_LENGTH,
    SAMPLE_RATE,
    io,
    librosa,
    mo,
    np,
    plt,
    sf,
    spectral_comparison,
):
    def audio_player(artifact):
        _buffer = io.BytesIO()
        sf.write(_buffer, artifact.audio.T, artifact.sr, format="WAV", subtype="FLOAT")
        _buffer.seek(0)
        return mo.audio(_buffer)


    def waveform_figure(artifact):
        _mono = artifact.mono()
        _step = max(1, _mono.audio.shape[-1] // 20000)
        _time = np.arange(0, _mono.audio.shape[-1], _step) / _mono.sr
        _fig, _axis = plt.subplots(figsize=(14, 2.5), layout="constrained")
        _axis.plot(_time, _mono.audio[::_step], linewidth=0.5)
        _axis.set(title="Fragmento cargado · mono a 22.05 kHz", xlabel="Tiempo (s)",
                  ylabel="Amplitud")
        _axis.grid(alpha=0.2)
        return _fig


    def compare_audio(artifacts, reference_audio, representation):
        _comparison = spectral_comparison(artifacts, reference_audio, representation)
        _mono = [artifact.mono() for artifact in artifacts.values()]
        _peak = max(1e-4, max(float(np.max(np.abs(a.audio))) for a in _mono))
        _fig, _axes = plt.subplots(len(_mono), 2, figsize=(14, 3 * len(_mono)), squeeze=False,
                                   sharex=True, sharey="col", layout="constrained")
        _image = None
        for _row, (_name, _artifact) in enumerate(zip(artifacts, _mono)):
            # Diezmado solo para dibujar: la inferencia usa todas las muestras.
            _step = max(1, _artifact.audio.shape[-1] // 20000)
            _time = np.arange(0, _artifact.audio.shape[-1], _step) / _artifact.sr
            _axes[_row, 0].plot(_time, _artifact.audio[::_step], linewidth=0.5)
            _axes[_row, 0].set(title=f"{_name} · forma de onda", ylim=(-_peak, _peak),
                               ylabel="Amplitud", xlabel="Tiempo (s)")
            _image = librosa.display.specshow(
                _comparison.matrices[_name], sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                x_axis="time", y_axis=_comparison.y_axis, fmin=0, fmax=SAMPLE_RATE / 2,
                cmap="magma", vmin=_comparison.vmin, vmax=_comparison.vmax,
                ax=_axes[_row, 1])
            _axes[_row, 1].set_title(f"{_name} · {representation}")
            _axes[_row, 0].set_xlim(0, reference_audio.duration)
        if _image is not None:
            _fig.colorbar(_image, ax=list(_axes[:, 1]), label=_comparison.label)
        return _fig


    def piano_roll_figure(events_by_name, duration, title):
        _fig, _axes = plt.subplots(len(events_by_name), 1, figsize=(14, 2.6 * len(events_by_name)),
                                   squeeze=False, sharex=True, sharey=True, layout="constrained")
        for _row, (_name, _events) in enumerate(events_by_name.items()):
            _axis = _axes[_row, 0]
            for _onset, _offset, _pitch, *_ in _events:
                _axis.add_patch(plt.Rectangle((_onset, _pitch - 0.4), _offset - _onset, 0.8,
                                              facecolor=f"C{_row % 10}", alpha=0.7))
            _axis.set(title=f"{title} · {_name} · {len(_events)} notas", xlim=(0, duration),
                      ylim=(20, 109), ylabel="Nota MIDI", xlabel="Tiempo en el fragmento (s)")
            _axis.grid(alpha=0.2)
        return _fig


    return audio_player, compare_audio, piano_roll_figure, waveform_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 08 · Ejemplo del pipeline completo

    Recorrido lineal de una mezcla por todas las etapas de AcusPy:

    ```text
    audio original
      → HT-Demucs (vocals / drums / bass / other)
      → copia mono a 22.05 kHz
      → Pre-DSP por variante
      → Basic Pitch
      → Post-DSP de notas
      → MIDI y benchmark opcional
    ```

    Cada control está siempre disponible. En modo script se usa el track de
    demostración y se ejecuta todo sin interacción; al abrir el notebook,
    selecciona el audio y pulsa **Separar con HT-Demucs**. La primera
    separación puede descargar los pesos de Demucs. Empieza con **6–10 s**.
    """)
    return


@app.cell
def _(mo, project_root):
    demo_track = project_root / "data/babyslakh_16k/Track00001/mix.wav"
    auto_run = mo.app_meta().mode == "script"

    audio_picker = mo.ui.file_browser(
        initial_path=project_root / "data" if (project_root / "data").is_dir() else project_root,
        filetypes=[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"],
        selection_mode="file",
        multiple=False,
        restrict_navigation=False,
        label="Seleccionar audio")
    offset_control = mo.ui.number(start=0, value=0, label="Inicio (segundos)")
    duration_control = mo.ui.number(start=1, stop=120, value=8,
                                    label="Duración del fragmento (segundos)")

    _demo_note = (mo.md(f"En modo script se usa `{demo_track.relative_to(project_root)}` "
                        f"con los valores iniciales.")
                  if demo_track.is_file() else mo.md("Selecciona un audio del proyecto."))
    mo.vstack([_demo_note, audio_picker, mo.hstack([offset_control, duration_control], wrap=True)])
    return audio_picker, auto_run, demo_track, duration_control, offset_control


@app.cell
def _(
    AudioArtifact,
    Path,
    audio_picker,
    auto_run,
    demo_track,
    duration_control,
    librosa,
    mo,
    offset_control,
):
    _selected = audio_picker.value
    if _selected:
        source_path = Path(_selected[0].path).expanduser()
    elif auto_run and demo_track.is_file():
        source_path = demo_track
    else:
        mo.stop(True, mo.md("Selecciona un audio en el explorador para comenzar."))
    mo.stop(not source_path.is_file(), mo.md(f"No existe el archivo: `{source_path}`"))

    input_signal, input_sr = librosa.load(
        str(source_path), sr=None, mono=False,
        offset=offset_control.value, duration=duration_control.value,
    )
    mo.stop(input_signal.size == 0, mo.md("El fragmento seleccionado está vacío."))
    input_artifact = AudioArtifact(input_signal, input_sr)
    return input_artifact, source_path


@app.cell
def _(audio_player, input_artifact, mo, source_path, waveform_figure):
    _channels = "estéreo" if input_artifact.audio.ndim == 2 else "mono"
    mo.vstack([
        mo.md(f"**{source_path.name}** · {input_artifact.sr} Hz · {_channels} · "
              f"{input_artifact.duration:.2f} s"),
        audio_player(input_artifact),
        waveform_figure(input_artifact),
    ])
    return


@app.cell
def _(available_devices, mo):
    device_selector = mo.ui.dropdown(
        options=available_devices(), value="cpu",
        label="Dispositivo Demucs (mps: GPU Apple · cuda: NVIDIA)")
    separate_button = mo.ui.run_button(label="Separar con HT-Demucs")
    mo.hstack([device_selector, separate_button], align="end", wrap=True)
    return device_selector, separate_button


@app.cell
def _(
    DemucsTranscriber,
    auto_run,
    device_selector,
    input_artifact,
    mo,
    separate_button,
):
    mo.stop(not (auto_run or separate_button.value),
            mo.md("Pulsa **Separar con HT-Demucs** para ejecutar la separación."))
    with mo.status.spinner(title="Separando el fragmento con HT-Demucs…"):
        engine = DemucsTranscriber(device=device_selector.value)
        separation = engine.separate(input_artifact.audio, sr=input_artifact.sr)
    return engine, separation


@app.cell
def _(audio_player, mo, separation):
    mo.vstack([
        mo.md(f"Modelo **{separation.model_name}** · salida estéreo a "
              f"{separation.demucs_input.sr} Hz · {separation.demucs_input.duration:.2f} s"),
        mo.ui.table([
            {"stem": _name, "Hz": _artifact.sr, "segundos": round(_artifact.duration, 3)}
            for _name, _artifact in separation.stems.items()
        ], selection=None),
        mo.hstack([
            mo.vstack([mo.md(f"**{_name}**"), audio_player(_artifact)])
            for _name, _artifact in separation.stems.items()
        ], wrap=True),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Pre-DSP

    Cada variante parte de una copia del stem mono a 22.05 kHz. Cada efecto es
    una etapa recuperable; cambiar un control recalcula el DSP sin repetir Demucs.
    """)
    return


@app.cell
def _(mo, separation):
    stem_selector = mo.ui.dropdown(options=list(separation.stems), value="bass",
                                    label="Stem")
    hp_control = mo.ui.slider(20, 1000, step=5, value=35, show_value=True,
                             label="Paso alto (Hz)")
    lp_control = mo.ui.slider(1100, 10000, step=100, value=5000, show_value=True,
                             label="Paso bajo (Hz)")
    gate_control = mo.ui.slider(-80, -10, step=1, value=-42, show_value=True,
                               label="Noise gate (dB)")
    mo.vstack([
        stem_selector,
        mo.hstack([hp_control, lp_control, gate_control], wrap=True),
    ])
    return gate_control, hp_control, lp_control, stem_selector


@app.cell
def _(
    AudioPreProcessor,
    DSPVariant,
    HighpassFilter,
    LowpassFilter,
    NoiseGate,
    engine,
    gate_control,
    hp_control,
    lp_control,
    separation,
    stem_selector,
):
    def _make_stages(_with_gate):
        # Instancias nuevas por rama: Pedalboard puede conservar estado interno.
        _stages = [
            AudioPreProcessor(name="highpass",
                              board=[HighpassFilter(cutoff_frequency_hz=hp_control.value)]),
            AudioPreProcessor(name="lowpass",
                              board=[LowpassFilter(cutoff_frequency_hz=lp_control.value)]),
        ]
        if _with_gate:
            _stages.append(AudioPreProcessor(name="gate", board=[
                NoiseGate(threshold_db=gate_control.value, ratio=6.0)]))
        return _stages


    variant_configs = {
        "raw": DSPVariant("raw"),
        "filtered": DSPVariant("filtered", _make_stages(True)),
        "alternative": DSPVariant("alternative", _make_stages(False)),
    }
    dsp_results = engine.process_variants(separation, {
        stem_selector.value: [variant_configs[_name] for _name in ("raw", "filtered", "alternative")],
    })
    return (dsp_results,)


@app.cell
def _(dsp_results, mo, stem_selector):
    stage_variant = mo.ui.dropdown(options=list(dsp_results[stem_selector.value]),
                                   value="filtered",
                                   label="Rama para inspeccionar etapas")
    stage_variant
    return (stage_variant,)


@app.cell
def _(audio_player, dsp_results, mo, stage_variant, stem_selector):
    _stage_artifacts = dsp_results[stem_selector.value][stage_variant.value].stages
    mo.vstack([
        mo.hstack([mo.md(f"**{_name}** · {_artifact.sr} Hz · {_artifact.duration:.2f} s"),
                   audio_player(_artifact)], widths=[2, 5], wrap=True)
        for _name, _artifact in _stage_artifacts.items()
    ])
    return


@app.cell
def _(
    compare_audio,
    dsp_results,
    mo,
    separation,
    stage_variant,
    stem_selector,
):
    _stage_figure = compare_audio(
        dsp_results[stem_selector.value][stage_variant.value].stages,
        separation.stems[stem_selector.value], "log-Mel")
    mo.vstack([mo.md("### Etapas DSP de la rama · referencia: stem sin DSP"), _stage_figure])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Procesamiento Basic Pitch

    La transcripción reutiliza el modelo ya cargado. Cambiar el DSP invalida
    los MIDI anteriores; los gráficos espectrales no afectan a la inferencia.
    """)
    return


@app.cell
def _(mo):
    representation_selector = mo.ui.dropdown(
        options=["STFT", "Mel", "log-Mel"], value="log-Mel",
        label="Representación espectral")
    mo.vstack([
        mo.md("""La representación es una ayuda visual: no modifica el audio que recibe
        Basic Pitch ni mejora por sí sola las métricas."""),
        representation_selector,
    ])
    return (representation_selector,)


@app.cell
def _(
    compare_audio,
    dsp_results,
    representation_selector,
    separation,
    stem_selector,
):
    _variant_artifacts = {_name: _result.final_audio
                         for _name, _result in dsp_results[stem_selector.value].items()}
    _variant_figure = compare_audio(
        _variant_artifacts, separation.stems[stem_selector.value], representation_selector.value)
    _variant_figure
    return


@app.cell
def _(mo):
    transcribe_button = mo.ui.run_button(label="Transcribir variantes con Basic Pitch")
    transcribe_button
    return (transcribe_button,)


@app.cell
def _(auto_run, dsp_results, engine, mo, transcribe_button):
    mo.stop(not (auto_run or transcribe_button.value),
            mo.md("Pulsa **Transcribir variantes** para ejecutar Basic Pitch con el DSP actual."))
    with mo.status.spinner(title="Transcribiendo las variantes…"):
        transcriptions = engine.transcribe_variants(dsp_results)
    return (transcriptions,)


@app.cell
def _(mo, stem_selector, transcriptions):
    mo.ui.table([
        {"variante": _name,
         "notas del modelo": len(_result.transcription["raw_note_events"]),
         "fin del MIDI (s)": round(_result.transcription["raw_midi"].get_end_time(), 3)}
        for _name, _result in transcriptions[stem_selector.value].items()
    ], selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Post-DSP

    El Post-DSP filtra eventos de notas por duración mínima, tesitura o
    velocidad. Solo `filtered` lo aplica; comparar ambos MIDI muestra qué
    notas elimina el filtro.
    """)
    return


@app.cell
def _(mo):
    min_duration_control = mo.ui.slider(0.01, 0.3, step=0.01, value=0.05,
                                        show_value=True, label="Duración mínima de nota (s)")
    pitch_low_control = mo.ui.slider(21, 108, step=1, value=28, show_value=True,
                                     label="Nota MIDI mínima")
    pitch_high_control = mo.ui.slider(21, 108, step=1, value=67, show_value=True,
                                      label="Nota MIDI máxima")
    min_velocity_control = mo.ui.slider(0.0, 1.0, step=0.05, value=0.0, show_value=True,
                                        label="Velocidad mínima (0: sin filtro)")
    mo.vstack([
        mo.md("""El Post-DSP no repite la inferencia: filtra los eventos de notas del
        modelo y reconstruye el MIDI. Solo `filtered` lo aplica."""),
        mo.hstack([min_duration_control, pitch_low_control, pitch_high_control,
                   min_velocity_control], wrap=True),
    ])
    return (
        min_duration_control,
        min_velocity_control,
        pitch_high_control,
        pitch_low_control,
    )


@app.cell
def _(
    NotePostProcessor,
    min_duration_control,
    min_velocity_control,
    pitch_high_control,
    pitch_low_control,
    replace,
    stem_selector,
    transcriptions,
):
    _postprocessor = NotePostProcessor(
        pitch_range=(pitch_low_control.value, pitch_high_control.value),
        min_duration_sec=min_duration_control.value,
        min_velocity=min_velocity_control.value,
        name="demo_post_dsp")
    _filtered_result = transcriptions[stem_selector.value]["filtered"]
    _filtered_events, _filtered_midi = _postprocessor.process(
        _filtered_result.transcription["raw_note_events"],
        base_midi=_filtered_result.transcription["raw_midi"])
    post_dsp_results = {
        **transcriptions[stem_selector.value],
        "filtered": replace(
            _filtered_result,
            note_postprocessor=_postprocessor,
            transcription={**_filtered_result.transcription,
                           "midi": _filtered_midi,
                           "note_events": _filtered_events,
                           "applied_post_dsp": True},
        ),
    }
    return (post_dsp_results,)


@app.cell
def _(mo, piano_roll_figure, post_dsp_results):
    _filtered_result = post_dsp_results["filtered"]
    _filtered_figure = piano_roll_figure(
        {"crudo": _filtered_result.transcription["raw_note_events"],
         "post-DSP": _filtered_result.transcription["note_events"]},
        _filtered_result.final_audio.duration, "filtered")
    mo.vstack([
        _filtered_figure,
        mo.ui.table([
            {"salida": "crudo", "notas": len(_filtered_result.transcription["raw_note_events"])},
            {"salida": "post-DSP", "notas": len(_filtered_result.transcription["note_events"])},
        ], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Benchmark

    La referencia debe corresponder al stem evaluado y estar alineada con el
    audio. Para `bass`, selecciona los MIDI del bajo; `all_src.mid` solo sirve
    como referencia aproximada de la mezcla completa. En modo script se usa ese
    archivo cuando está disponible.
    """)
    return


@app.cell
def _(mo, source_path, stem_selector):
    mo.stop(stem_selector.value == "drums", mo.md("Drums queda fuera del benchmark de notas."))
    _midi_dir = source_path.parent / "MIDI"
    reference_picker = mo.ui.file_browser(
        initial_path=_midi_dir if _midi_dir.is_dir() else source_path.parent,
        filetypes=[".mid", ".midi"], selection_mode="file", multiple=True,
        restrict_navigation=False,
        label=f"MIDI de referencia para {stem_selector.value} (selección explícita)")
    benchmark_button = mo.ui.run_button(label="Evaluar benchmark")
    mo.vstack([reference_picker, benchmark_button])
    return benchmark_button, reference_picker


@app.cell
def _(
    Path,
    auto_run,
    benchmark_button,
    benchmark_variants,
    load_reference_midis,
    mo,
    offset_control,
    post_dsp_results,
    reference_picker,
    separation,
    source_path,
    stem_selector,
):
    mo.stop(stem_selector.value == "drums", mo.md("Drums queda fuera del benchmark de notas."))
    if reference_picker.value:
        reference_paths = [Path(_entry.path) for _entry in reference_picker.value]
    elif auto_run and (source_path.parent / "all_src.mid").is_file():
        reference_paths = [source_path.parent / "all_src.mid"]
    else:
        reference_paths = []
    mo.stop(not reference_paths,
            mo.md("Selecciona uno o varios MIDI de referencia alineados con el audio."))
    mo.stop(not (auto_run or benchmark_button.value),
            mo.md("Pulsa **Evaluar benchmark** con la referencia seleccionada."))

    benchmark_duration = separation.input_audio.duration
    reference_midi = load_reference_midis(
        reference_paths, start=offset_control.value, duration=benchmark_duration)
    benchmark_rows = benchmark_variants(post_dsp_results, reference_midi, benchmark_duration)

    _messages = [mo.md(
        f"**Stem:** {stem_selector.value} · **Fragmento:** "
        f"{offset_control.value:.3f}–{offset_control.value + benchmark_duration:.3f} s · "
        f"**Referencias:** {', '.join(_path.name for _path in reference_paths)}")]
    if benchmark_rows and benchmark_rows[0]["estado"] == "sin referencia evaluable":
        _messages.append(mo.md("**sin referencia evaluable**: no hay notas no percusivas "
                               "en este fragmento; las métricas quedan sin calcular."))
    if auto_run:
        _messages.append(mo.md("En modo script la referencia es `all_src.mid`, aproximada "
                               "para un único stem."))
    _display_rows = [
        {_key: (round(_value, 4) if isinstance(_value, float) else _value)
         for _key, _value in _row.items()}
        for _row in benchmark_rows
    ]
    _messages.append(mo.ui.table(_display_rows, selection=None, show_download=True))
    mo.vstack(_messages)
    return


if __name__ == "__main__":
    app.run()
