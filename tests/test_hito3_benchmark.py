"""Corrida reanudable del notebook 01, con un runner falso (sin Demucs ni Basic Pitch)."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.hito3 import CONDITIONS, RESULT_COLUMNS, TrackGroup, read_rows
from src.hito3_benchmark import (
    META_NAME, merge_runs, read_meta, run_benchmark, track_csv_path, track_done,
)


def group(track_id):
    return TrackGroup(track_id, Path("/no/existe") / track_id, "bass",
                      stem_ids=["S00"], inst_classes=["Bass"], programs=["Electric Bass (finger)"])


class FakeRunner:
    """Devuelve filas válidas; falla en los tracks de ``fail`` mientras sigan ahí."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls = []

    def __call__(self, tg, start, duration, model):
        self.calls.append(tg.track_id)
        if tg.track_id in self.fail:
            raise RuntimeError("stem vacío,\nsin notas")
        return [{
            "track_id": tg.track_id, "grupo": tg.group, "stems_ref": "S00", "inst_class": "Bass",
            "modelo_demucs": model, "condicion": key, "pre_dsp": int(pre), "post_dsp": int(post),
            "inicio_s": start, "duracion_s": duration, "ref_transpose": -12,
            "P_no": 0.5, "R_no": 0.5, "F_no": 0.5 + 0.1 * pre, "P": 0.4, "R": 0.4, "F": 0.4,
            "Acc": 0.3, "n_ref": 10, "n_pred": 10, "estado": "ok",
        } for key, _, pre, post in CONDITIONS]


class RunBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.out = self.root / "benchmark_bajo"
        self.groups = [group("Track00003"), group("Track00001"), group("Track00002")]

    def tearDown(self):
        self._tmp.cleanup()

    def test_un_csv_por_track_sin_temporales(self):
        status = run_benchmark(self.groups, FakeRunner(), self.out, duration=30.0)
        self.assertEqual(set(status.values()), {"ok"})
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         sorted([META_NAME, "Track00001.csv", "Track00002.csv", "Track00003.csv"]))
        with track_csv_path(self.out, "Track00001").open(encoding="utf-8") as fh:
            self.assertEqual(tuple(csv.DictReader(fh).fieldnames), RESULT_COLUMNS)

    def test_track_que_falla_escribe_error_y_la_corrida_sigue(self):
        status = run_benchmark(self.groups, FakeRunner(fail={"Track00001"}), self.out, duration=30.0)
        self.assertEqual(status["Track00002"], "ok")
        self.assertEqual(status["Track00003"], "ok")
        rows = read_rows(track_csv_path(self.out, "Track00001"))
        self.assertEqual([r["condicion"] for r in rows], [k for k, *_ in CONDITIONS])
        for r in rows:
            self.assertEqual(r["estado"], "error: RuntimeError: stem vacío, sin notas")
            self.assertIsNone(r["F_no"])
            self.assertEqual(r["duracion_s"], 30.0)
        self.assertFalse(track_done(track_csv_path(self.out, "Track00001")))

    def test_al_retomar_omite_hechos_y_reintenta_errores(self):
        run_benchmark(self.groups, FakeRunner(fail={"Track00001"}), self.out)
        runner = FakeRunner()
        status = run_benchmark(self.groups, runner, self.out)
        self.assertEqual(runner.calls, ["Track00001"])
        self.assertEqual(status, {"Track00003": "omitido", "Track00001": "ok", "Track00002": "omitido"})
        self.assertTrue(track_done(track_csv_path(self.out, "Track00001")))

    def test_csv_incompleto_no_cuenta_como_hecho(self):
        run_benchmark(self.groups[:1], FakeRunner(), self.out)
        path = track_csv_path(self.out, "Track00003")
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")
        self.assertFalse(track_done(path))

    def test_meta_guarda_dispositivo_versiones_y_tiempos(self):
        run_benchmark(self.groups, FakeRunner(), self.out, duration=30.0, device="mps")
        meta = read_meta(self.out)
        self.assertEqual(meta["dispositivo"], "mps")
        self.assertEqual(set(meta["versiones"]), {"demucs", "basic-pitch", "torch"})
        self.assertEqual(set(meta["segundos_por_track"]), {g.track_id for g in self.groups})
        self.assertEqual(meta["config"]["duracion_s"], 30.0)
        self.assertIn("fecha", meta)
        json.loads((self.out / META_NAME).read_text(encoding="utf-8"))

    def test_rechaza_retomar_con_otra_configuracion(self):
        run_benchmark(self.groups, FakeRunner(), self.out, duration=30.0)
        with self.assertRaises(ValueError):
            run_benchmark(self.groups, FakeRunner(), self.out, duration=None)

    def test_merge_une_y_ordena(self):
        run_benchmark(self.groups, FakeRunner(fail={"Track00002"}), self.out)
        merged = merge_runs(self.out, self.root / "benchmark_bajo.csv")
        rows = read_rows(merged)
        self.assertEqual(len(rows), 12)
        self.assertEqual([r["track_id"] for r in rows[::4]], ["Track00001", "Track00002", "Track00003"])
        self.assertEqual([r["condicion"] for r in rows[:4]], [k for k, *_ in CONDITIONS])
        self.assertTrue(rows[4]["estado"].startswith("error"))
        self.assertFalse((self.root / "benchmark_bajo.csv.tmp").exists())

    def test_merge_rechaza_csv_fuera_del_contrato(self):
        self.out.mkdir()
        (self.out / "otro.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            merge_runs(self.out, self.root / "benchmark_bajo.csv")


if __name__ == "__main__":
    unittest.main()
