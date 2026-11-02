# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.25.1,<0.26",
#     "numpy<2",
#     "librosa",
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
    from pathlib import Path

    import marimo as mo

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")

    from src import DemucsTranscriber, available_devices, get_audio_pipeline, get_note_pipeline
    from src.hito3 import (
        CATALOG_COLUMNS,
        CONDITIONS,
        DATA_DIR,
        RESULT_COLUMNS,
        RESULTS_DIR,
        list_track_groups,
        reference_transpose,
        run_conditions,
        write_rows,
    )

    return (
        CATALOG_COLUMNS,
        CONDITIONS,
        DATA_DIR,
        DemucsTranscriber,
        RESULTS_DIR,
        RESULT_COLUMNS,
        available_devices,
        get_audio_pipeline,
        get_note_pipeline,
        list_track_groups,
        mo,
        project_root,
        reference_transpose,
        run_conditions,
        write_rows,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 00 · Setup común del Hito 3

    Fija lo que comparten los notebooks 01–04 para que sus resultados sean comparables:

    1. **Catálogo:** qué tracks de BabySlakh tienen bajo y cuáles se pueden evaluar.
    2. **Condiciones 2×2:** PRE-DSP on/off × POST-DSP on/off.
    3. **Presets DSP congelados:** los que se usan como punto de partida del benchmark.
    4. **Formato de resultados:** columnas fijas del CSV que escribe cada notebook.
    5. **Prueba de humo:** corre las 4 condiciones en un fragmento corto para validar el entorno.

    La lógica vive en `src/hito3.py`; este notebook solo la muestra y genera el catálogo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Catálogo de tracks con bajo

    La unidad de evaluación es **track × grupo Demucs**. Demucs entrega un solo stem `bass`
    por track, así que la referencia une los MIDI de todos los bajos de ese track. Un track
    solo es evaluable si **todos** sus bajos tienen MIDI: con una referencia parcial, las notas
    correctas del bajo faltante contarían como notas fantasma.
    """)
    return


@app.cell
def _(CATALOG_COLUMNS, DATA_DIR, RESULTS_DIR, list_track_groups, mo, write_rows):
    mo.stop(not DATA_DIR.is_dir(), mo.md(f"No encuentro BabySlakh en `{DATA_DIR}`. Revisa el Readme."))
    bass_groups = list_track_groups("bass")
    evaluable_bass = [g for g in bass_groups if g.evaluable]
    catalog_path = write_rows([g.catalog_row() for g in bass_groups],
                              RESULTS_DIR / "catalogo_bajo.csv", CATALOG_COLUMNS)
    _dobles = [g.track_id for g in evaluable_bass if len(g.stem_ids) > 1]
    _excluidos = [f"{g.track_id} (sin MIDI: {', '.join(g.missing_midi)})" for g in bass_groups if not g.evaluable]
    mo.vstack([
        mo.md(
            f"**{len(bass_groups)}** tracks con bajo · **{len(evaluable_bass)}** evaluables · "
            f"con dos bajos: {', '.join(_dobles) or 'ninguno'} · "
            f"excluidos: {', '.join(_excluidos) or 'ninguno'}.  \n"
            f"Catálogo guardado en `{catalog_path.relative_to(RESULTS_DIR.parents[1])}`."
        ),
        mo.ui.table([g.catalog_row() for g in bass_groups], selection=None, page_size=25),
    ])
    return (evaluable_bass,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Para el notebook 04 (htdemucs_6s)

    Con `htdemucs_6s`, guitarra y piano salen como grupos propios. Mismo criterio de evaluable.
    """)
    return


@app.cell
def _(CATALOG_COLUMNS, RESULTS_DIR, list_track_groups, mo, write_rows):
    _resumen = []
    for _grupo in ("bass", "guitar", "piano"):
        _gs = list_track_groups(_grupo, model="htdemucs_6s")
        write_rows([g.catalog_row() for g in _gs], RESULTS_DIR / f"catalogo_{_grupo}_6s.csv", CATALOG_COLUMNS)
        _resumen.append({
            "Grupo": _grupo, "Tracks": len(_gs), "Evaluables": sum(g.evaluable for g in _gs),
            "Instrumentos por track (máx.)": max((len(g.stem_ids) for g in _gs), default=0),
        })
    mo.ui.table(_resumen, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Condiciones y formato de resultados

    Cada notebook escribe **una fila por track × grupo × condición** con `write_rows`, que
    rechaza columnas fuera del contrato. Los CSV van en `results/hito3/` y sí se versionan.
    """)
    return


@app.cell
def _(CONDITIONS, RESULT_COLUMNS, mo):
    mo.hstack([
        mo.ui.table(
            [{"Clave": k, "Condición": label, "PRE-DSP": int(pre), "POST-DSP": int(post)}
             for k, label, pre, post in CONDITIONS],
            selection=None, label="Condiciones",
        ),
        mo.ui.table([{"#": i, "Columna": c} for i, c in enumerate(RESULT_COLUMNS)],
                    selection=None, page_size=30, label="Columnas del CSV"),
    ], widths=[1, 1], align="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Presets DSP congelados (bajo)

    Son los de `src/transcriber.py`, los mismos del Hito 2. El benchmark 01 los usa tal cual;
    si el notebook 02 propone un POST-DSP nuevo, se agrega como preset con otro nombre y se
    re-corre solo la columna POST, sin tocar este.
    """)
    return


@app.cell
def _(get_audio_pipeline, get_note_pipeline, mo, reference_transpose):
    pre_bass = get_audio_pipeline("Bass")
    post_bass = get_note_pipeline("Bass")

    def _param(plugin):
        for attr in ("cutoff_frequency_hz", "threshold_db", "ratio"):
            if hasattr(plugin, attr):
                yield f"{attr} = {getattr(plugin, attr):g}"

    mo.ui.table(
        [{"Etapa": "PRE-DSP", "Componente": type(p).__name__, "Parámetros": ", ".join(_param(p))}
         for p in pre_bass.board]
        + [
            {"Etapa": "POST-DSP", "Componente": "Tesitura (MIDI)", "Parámetros": str(post_bass.pitch_range)},
            {"Etapa": "POST-DSP", "Componente": "Duración mínima", "Parámetros": f"{post_bass.min_duration_sec} s"},
            {"Etapa": "POST-DSP", "Componente": "Velocidad mínima", "Parámetros": str(post_bass.min_velocity)},
            {"Etapa": "Referencia", "Componente": "Transposición", "Parámetros": f"{reference_transpose('bass'):+d} semitonos"},
        ],
        selection=None,
    )
    return post_bass, pre_bass


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Prueba de humo

    Corre las 4 condiciones sobre el primer track evaluable, en un fragmento corto. Sirve para
    comprobar que Demucs, Basic Pitch y la evaluación funcionan en esta máquina antes de lanzar
    el 01 completo. No guarda nada. Como script: `python notebooks/hito_03/00_setup_comun.py -- --smoke`.
    """)
    return


@app.cell
def _(available_devices, evaluable_bass, mo):
    smoke_track = mo.ui.dropdown(options=[g.track_id for g in evaluable_bass],
                                 value=evaluable_bass[0].track_id, label="Track")
    smoke_duration = mo.ui.number(start=5, stop=60, value=15, label="Duración (s)")
    smoke_device = mo.ui.dropdown(options=available_devices(), value="cpu", label="Dispositivo Demucs")
    smoke_button = mo.ui.run_button(label="Correr prueba de humo")
    mo.hstack([smoke_track, smoke_duration, smoke_device, smoke_button], wrap=True, justify="start")
    return smoke_button, smoke_device, smoke_duration, smoke_track


@app.cell
def _(
    DemucsTranscriber,
    evaluable_bass,
    mo,
    post_bass,
    pre_bass,
    run_conditions,
    smoke_button,
    smoke_device,
    smoke_duration,
    smoke_track,
):
    _cli = mo.app_meta().mode == "script" and "smoke" in mo.cli_args()
    mo.stop(not (smoke_button.value or _cli), mo.md("Pulsa **Correr prueba de humo** cuando quieras validar el entorno."))
    _tg = next(g for g in evaluable_bass if g.track_id == smoke_track.value)
    with mo.status.spinner(title=f"Separando y transcribiendo {_tg.track_id}…"):
        _engine = DemucsTranscriber(device=smoke_device.value)
        smoke_rows = run_conditions(_tg, _engine, pre_bass, post_bass, start=0.0,
                                    duration=float(smoke_duration.value))
    if _cli:
        for _r in smoke_rows:
            print(f"{_r['track_id']} {_r['condicion']:8s} F_no={_r['F_no']} n_ref={_r['n_ref']} n_pred={_r['n_pred']}")
    mo.ui.table([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in smoke_rows],
                selection=None)
    return (smoke_rows,)


if __name__ == "__main__":
    app.run()
