"""Benchmark 2×2 del bajo en BabySlakh (notebook 01 del Hito 3).

La corrida es reanudable: cada track escribe su propio CSV en ``<out_dir>/<track>.csv``
de forma atómica, así un corte (Patagón, batería, Ctrl+C) solo pierde el track en curso.
Un track que falla deja sus 4 filas con ``estado = "error: …"`` y se reintenta al retomar.
``merge_runs`` une los CSV por track en el CSV final del contrato.

No carga modelos por sí mismo: recibe un ``runner`` (``make_runner`` en uso real, uno falso
en los tests) con la firma ``runner(tg, start, duration, model) -> list[dict]``.
"""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from .hito3 import CONDITIONS, METRICS, RESULT_COLUMNS, TrackGroup, reference_transpose, run_conditions, write_rows

META_NAME = "_meta.json"
VERSIONED_PACKAGES = ("demucs", "basic-pitch", "torch")

Runner = Callable[[TrackGroup, float, Optional[float], str], list]


def make_runner(engine, pre_processor, post_processor,
                midi_dir: Optional[Union[str, Path]] = None) -> Runner:
    """Runner real: ``run_conditions`` con un motor y presets ya cargados."""
    def runner(tg, start, duration, model):
        return run_conditions(tg, engine, pre_processor, post_processor, start=start,
                              duration=duration, model=model, midi_dir=midi_dir)
    return runner


def track_csv_path(out_dir: Union[str, Path], track_id: str) -> Path:
    return Path(out_dir) / f"{track_id}.csv"


def write_rows_atomic(rows: Iterable[dict], path: Union[str, Path]) -> Path:
    """``write_rows`` a un temporal y ``os.replace``: el CSV final nunca queda a medias."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    write_rows(rows, tmp)
    os.replace(tmp, path)
    return path


def error_rows(tg: TrackGroup, exc: BaseException, start: float,
               duration: Optional[float], model: str) -> list[dict]:
    """Las 4 filas de un track fallido: metadatos completos, métricas vacías."""
    message = " ".join(f"{type(exc).__name__}: {exc}".split())
    return [{
        "track_id": tg.track_id, "grupo": tg.group, "stems_ref": "+".join(tg.stem_ids),
        "inst_class": "+".join(sorted(set(tg.inst_classes))), "modelo_demucs": model,
        "condicion": key, "pre_dsp": int(pre_on), "post_dsp": int(post_on),
        "inicio_s": start, "duracion_s": duration, "ref_transpose": reference_transpose(tg.group),
        **dict.fromkeys(METRICS), "n_ref": None, "n_pred": None, "estado": f"error: {message}",
    } for key, _, pre_on, post_on in CONDITIONS]


def track_done(path: Union[str, Path]) -> bool:
    """Un track está hecho si su CSV existe, trae las 4 condiciones y ninguna con error."""
    path = Path(path)
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return ({r["condicion"] for r in rows} == {k for k, *_ in CONDITIONS}
            and not any(r["estado"].startswith("error") for r in rows))


def package_versions(packages: Iterable[str] = VERSIONED_PACKAGES) -> dict:
    out = {}
    for name in packages:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def read_meta(out_dir: Union[str, Path]) -> dict:
    path = Path(out_dir) / META_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_meta(out_dir: Path, meta: dict) -> None:
    path = out_dir / META_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_benchmark(groups: Iterable[TrackGroup], runner: Runner, out_dir: Union[str, Path],
                  start: float = 0.0, duration: Optional[float] = None,
                  model: str = "htdemucs", device: str = "cpu",
                  log: Optional[Callable[[str], None]] = None) -> dict:
    """Corre los tracks pendientes y devuelve ``{track_id: "ok" | "omitido" | "error: …"}``.

    Omite los tracks ya hechos (``track_done``) y reintenta los que quedaron con error.
    Si ``out_dir`` ya tiene una corrida con otra configuración (ventana, modelo o grupo),
    falla en vez de mezclar resultados no comparables.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = list(groups)
    log = log or (lambda msg: None)

    config = {"grupo": groups[0].group if groups else None, "modelo_demucs": model,
              "inicio_s": start, "duracion_s": duration}
    meta = read_meta(out_dir)
    if meta.get("config") not in (None, config):
        raise ValueError(f"{out_dir} tiene una corrida con otra configuración "
                         f"({meta['config']} ≠ {config}); usa otra carpeta o bórrala.")
    meta.update(config=config, dispositivo=device, versiones=package_versions())
    meta.setdefault("segundos_por_track", {})

    status = {}
    for i, tg in enumerate(groups, 1):
        path = track_csv_path(out_dir, tg.track_id)
        if track_done(path):
            status[tg.track_id] = "omitido"
            log(f"[{i}/{len(groups)}] {tg.track_id} ya está hecho, lo omito")
            continue
        t0 = time.perf_counter()
        try:
            rows = runner(tg, start, duration, model)
            status[tg.track_id] = "ok"
        except Exception as exc:  # un track roto no debe detener la corrida
            rows = error_rows(tg, exc, start, duration, model)
            status[tg.track_id] = rows[0]["estado"]
        seconds = round(time.perf_counter() - t0, 2)
        write_rows_atomic(rows, path)
        meta["segundos_por_track"][tg.track_id] = seconds
        meta["fecha"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_meta(out_dir, meta)
        log(f"[{i}/{len(groups)}] {tg.track_id} {status[tg.track_id]} ({seconds:.1f} s)")
    return status


def merge_runs(out_dir: Union[str, Path], out_path: Union[str, Path]) -> Path:
    """Une los CSV por track en uno solo, ordenado por track y condición del 2×2."""
    order = {k: i for i, (k, *_) in enumerate(CONDITIONS)}
    rows = []
    for path in sorted(Path(out_dir).glob("*.csv")):
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
                raise ValueError(f"{path.name} no sigue RESULT_COLUMNS.")
            rows.extend(reader)
    rows.sort(key=lambda r: (r["track_id"], r["grupo"], order.get(r["condicion"], len(order))))
    return write_rows_atomic(rows, out_path)
