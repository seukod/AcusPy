"""Contrato del Hito 3: catálogo, grupos Demucs y CSV, sin ejecutar modelos."""
import tempfile
import unittest
from pathlib import Path

from src.hito3 import (
    CONDITIONS, RESULT_COLUMNS, demucs_group, list_track_groups, read_rows,
    reference_transpose, write_rows,
)

META = """stems:
  S00: {inst_class: Bass, midi_program_name: Electric Bass (finger), is_drum: false}
  S01: {inst_class: Bass, midi_program_name: Slap Bass 1, is_drum: false}
  S02: {inst_class: Guitar, midi_program_name: Overdriven Guitar, is_drum: false}
  S03: {inst_class: Drums, midi_program_name: Drums, is_drum: true}
"""


def make_track(root, name, midis):
    track = Path(root) / name
    (track / "MIDI").mkdir(parents=True)
    (track / "metadata.yaml").write_text(META, encoding="utf-8")
    for stem in midis:
        (track / "MIDI" / f"{stem}.mid").write_bytes(b"")
    return track


class Hito3Tests(unittest.TestCase):
    def test_grupos_demucs(self):
        self.assertEqual(demucs_group("Bass", "Fretless Bass"), "bass")
        self.assertEqual(demucs_group("Guitar", "Overdriven Guitar"), "other")
        self.assertEqual(demucs_group("Guitar", "Overdriven Guitar", "htdemucs_6s"), "guitar")
        self.assertEqual(demucs_group("Piano", "Clavinet", "htdemucs_6s"), "piano")
        self.assertEqual(demucs_group("Synth Lead", "Voice Oohs"), "vocals")

    def test_transposicion_solo_bajo(self):
        self.assertEqual(reference_transpose("bass"), -12)
        self.assertEqual(reference_transpose("guitar"), 0)

    def test_catalogo_une_bajos_y_marca_faltantes(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_track(tmp, "Track00001", ["S00", "S01", "S02"])
            make_track(tmp, "Track00002", ["S00", "S02"])
            groups = list_track_groups("bass", tmp)
        self.assertEqual([g.track_id for g in groups], ["Track00001", "Track00002"])
        self.assertEqual(groups[0].stem_ids, ["S00", "S01"])
        self.assertTrue(groups[0].evaluable)
        self.assertEqual(groups[1].missing_midi, ["S01"])
        self.assertFalse(groups[1].evaluable)

    def test_drums_no_es_grupo_valido(self):
        with self.assertRaises(ValueError):
            list_track_groups("drums")

    def test_csv_ida_y_vuelta(self):
        row = {"track_id": "Track00001", "grupo": "bass", "condicion": "pre",
               "pre_dsp": 1, "post_dsp": 0, "F_no": 0.5, "n_ref": 10, "estado": "ok"}
        with tempfile.TemporaryDirectory() as tmp:
            path = write_rows([row], Path(tmp) / "r.csv")
            back = read_rows(path)[0]
        self.assertEqual(list(back), list(RESULT_COLUMNS))
        self.assertEqual(back["F_no"], 0.5)
        self.assertEqual(back["pre_dsp"], 1)
        self.assertIsNone(back["F"])

    def test_csv_rechaza_columnas_extra(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            write_rows([{"track_id": "x", "inventada": 1}], Path(tmp) / "r.csv")

    def test_condiciones_2x2(self):
        self.assertEqual({(pre, post) for _, _, pre, post in CONDITIONS},
                         {(False, False), (True, False), (False, True), (True, True)})


if __name__ == "__main__":
    unittest.main()
