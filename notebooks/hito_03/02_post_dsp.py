# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.25.1,<0.26",
#     "numpy<2",
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

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import warnings
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")

    from src import DemucsTranscriber, available_devices, get_audio_pipeline, get_note_pipeline
    from src.hito3 import (
        BASIC_PITCH_MIN_NOTE_S,
        NOTE_COLUMNS,
        POST_REASONS,
        RESULTS_DIR,
        audit_post_dsp,
        list_track_groups,
        transcribe_stem,
        write_rows,
    )

    return (
        BASIC_PITCH_MIN_NOTE_S,
        Counter,
        DemucsTranscriber,
        NOTE_COLUMNS,
        POST_REASONS,
        RESULTS_DIR,
        audit_post_dsp,
        available_devices,
        get_audio_pipeline,
        get_note_pipeline,
        list_track_groups,
        mo,
        plt,
        transcribe_stem,
        write_rows,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 02 · ¿Por qué el POST-DSP no actúa?

    En el Hito 2 las condiciones **post** y **raw** dieron el mismo F1. Este notebook mide
    cuánto actúa hoy el POST-DSP del bajo: para cada nota que entrega Basic Pitch, qué filtro
    del preset la eliminaría (`duracion`, `tesitura`, `velocidad`) o si pasa intacta (`ninguno`).

    Hipótesis de partida: Basic Pitch ya descarta las notas de menos de ~128 ms
    (`minimum_note_length`), así que un filtro de duración de 50 ms nunca encuentra a quién eliminar.
    """)
    return


@app.cell
def _(get_audio_pipeline, get_note_pipeline, list_track_groups, mo):
    pre_bass = get_audio_pipeline("Bass")
    post_bass = get_note_pipeline("Bass")
    evaluable_bass = [g for g in list_track_groups("bass") if g.evaluable]
    mo.md(
        f"Preset POST del bajo: tesitura **{post_bass.pitch_range}** (MIDI), "
        f"duración mínima **{post_bass.min_duration_sec * 1000:g} ms**, "
        f"velocidad mínima **{post_bass.min_velocity}**."
    )
    return evaluable_bass, post_bass, pre_bass


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Muestra

    Cada track se separa una vez con HT-Demucs y se transcribe sin y con PRE-DSP. Las notas
    se guardan en `results/hito3/post_dsp_notas.csv`. Como script:
    `python notebooks/hito_03/02_post_dsp.py -- --run`.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(available_devices, evaluable_bass, mo):
    track_picker = mo.ui.multiselect(options=[g.track_id for g in evaluable_bass],
                                     value=[g.track_id for g in evaluable_bass[:3]], label="Tracks")
    window_s = mo.ui.number(start=10, stop=120, value=30, label="Duración (s)")
    device = mo.ui.dropdown(options=available_devices(), value="cpu", label="Dispositivo Demucs")
    run_button = mo.ui.run_button(label="Transcribir y auditar")
    mo.hstack([track_picker, window_s, device, run_button], wrap=True, justify="start")
    return device, run_button, track_picker, window_s


@app.cell
def _():
    return


@app.cell
def _(
    DemucsTranscriber,
    NOTE_COLUMNS,
    RESULTS_DIR,
    audit_post_dsp,
    device,
    evaluable_bass,
    mo,
    post_bass,
    pre_bass,
    run_button,
    track_picker,
    transcribe_stem,
    window_s,
    write_rows,
):
    _cli = mo.app_meta().mode == "script" and "run" in mo.cli_args()
    mo.stop(not (run_button.value or _cli), mo.md("Pulsa **Transcribir y auditar** para generar las notas."))
    _engine = DemucsTranscriber(device=device.value)
    notas = []
    for _tg in mo.status.progress_bar([g for g in evaluable_bass if g.track_id in track_picker.value],
                                      title="Separando y transcribiendo"):
        _raw_t, _pre_t, _ = transcribe_stem(_tg, _engine, pre_bass, duration=float(window_s.value))
        notas += audit_post_dsp(_tg, {"raw": _raw_t, "pre": _pre_t}, post_bass)
    notas_path = write_rows(notas, RESULTS_DIR / "post_dsp_notas.csv", NOTE_COLUMNS)
    mo.md(f"**{len(notas)}** notas auditadas · guardadas en `results/hito3/{notas_path.name}`.")
    return (notas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Cuánto elimina el POST-DSP

    Una fila por track y fuente (`raw` = stem separado, `pre` = stem con PRE-DSP).
    """)
    return


@app.cell
def _(Counter, POST_REASONS, mo, notas):
    _conteo = Counter((n["track_id"], n["fuente"], n["motivo"]) for n in notas)
    _claves = sorted({(n["track_id"], n["fuente"]) for n in notas})
    resumen = []
    for _track, _fuente in _claves:
        _total = sum(_conteo[(_track, _fuente, m)] for m in POST_REASONS)
        _fila = {"Track": _track, "Fuente": _fuente, "Notas": _total}
        _fila.update({m: _conteo[(_track, _fuente, m)] for m in POST_REASONS if m != "ninguno"})
        _fila["% eliminadas"] = round(100 * (1 - _conteo[(_track, _fuente, "ninguno")] / _total), 1) if _total else 0.0
        resumen.append(_fila)
    mo.ui.table(resumen, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Las notas frente a los umbrales

    Si los umbrales del preset caen fuera de donde están las notas, el filtro no puede actuar.
    """)
    return


@app.cell
def _(BASIC_PITCH_MIN_NOTE_S, notas, plt, post_bass):
    # Paleta de referencia (dataviz): slot 1 = raw, slot 2 = pre; umbrales en tinta neutra.
    _colores = {"raw": "#2a78d6", "pre": "#eb6834"}
    _tinta, _tinta_suave = "#0b0b0b", "#52514e"

    fig_umbrales, (_ax_dur, _ax_pitch) = plt.subplots(1, 2, figsize=(11, 3.6))
    # Bins comunes a raw y pre (50 ms): con bins=40 cada fuente usaba su propio ancho y las alturas no se comparaban.
    _bins_ms = range(0, int(max(n["duracion_s"] for n in notas) * 1000) + 50, 50)
    for _fuente, _color in _colores.items():
        _sel = [n for n in notas if n["fuente"] == _fuente]
        _ax_dur.hist([n["duracion_s"] * 1000 for n in _sel], bins=_bins_ms, histtype="step",
                     linewidth=2, color=_color, label=_fuente)
        _ax_pitch.hist([n["pitch"] for n in _sel], bins=range(20, 90), histtype="step",
                       linewidth=2, color=_color, label=_fuente)

    _post_ms, _bp_ms = post_bass.min_duration_sec * 1000, BASIC_PITCH_MIN_NOTE_S * 1000
    _ax_dur.axvline(_post_ms, color=_tinta, linestyle="--", linewidth=1)
    _ax_dur.axvline(_bp_ms, color=_tinta_suave, linestyle=":", linewidth=1)
    _ax_dur.annotate(f"POST {_post_ms:g} ms", (_post_ms, 1), xycoords=("data", "axes fraction"),
                     xytext=(3, -12), textcoords="offset points", fontsize=8, color=_tinta)
    _ax_dur.annotate(f"Basic Pitch {_bp_ms:.0f} ms", (_bp_ms, 1), xycoords=("data", "axes fraction"),
                     xytext=(3, -24), textcoords="offset points", fontsize=8, color=_tinta_suave)
    _ax_dur.set(title="Duración de las notas", xlabel="ms", ylabel="Notas", xlim=(0, None))

    _p_min, _p_max = post_bass.pitch_range
    _ax_pitch.axvspan(_p_min, _p_max + 1, color=_tinta_suave, alpha=0.08, linewidth=0)
    _ax_pitch.annotate(f"tesitura POST {_p_min}–{_p_max}", (_p_min, 1), xycoords=("data", "axes fraction"),
                       xytext=(3, -12), textcoords="offset points", fontsize=8, color=_tinta_suave)
    _ax_pitch.set(title="Altura de las notas", xlabel="Pitch MIDI", ylabel="Notas")

    for _ax in (_ax_dur, _ax_pitch):
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
        _ax.set_axisbelow(True)
    _ax_pitch.legend(frameon=False, title="Fuente")
    fig_umbrales.tight_layout()
    fig_umbrales
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Qué observamos / qué decidimos

    _Completar después de correr la muestra (formato "pensábamos… observamos… decidimos…")._
    """)
    return


if __name__ == "__main__":
    app.run()
