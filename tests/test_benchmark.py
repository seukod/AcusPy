"""Casos musicales controlados para el benchmark, sin inferencia de modelos."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pretty_midi

from src.benchmark import (
    METRIC_KEYS, benchmark_variants, crop_midi, evaluate_prediction,
    load_reference_midis, midi_to_frame_frequencies, midi_to_intervals_and_pitches,
)


def make_midi(notes=(), is_drum=False):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0, is_drum=is_drum)
    for start, end, pitch in notes:
        instrument.notes.append(pretty_midi.Note(100, pitch, start, end))
    midi.instruments.append(instrument)
    return midi


def result(raw, final=None, stem="bass"):
    return SimpleNamespace(stem=stem, note_postprocessor=object() if final is not None else None,
                           transcription={"raw_midi": raw, "midi": final if final is not None else raw})


class BenchmarkTests(unittest.TestCase):
    def test_perfect_transcription(self):
        midi = make_midi([(0.1, 0.6, 60), (0.8, 1.5, 64)])
        metrics = evaluate_prediction(midi, midi, 2)
        for key in METRIC_KEYS:
            self.assertAlmostEqual(metrics[key], 1)
        self.assertEqual(metrics["Notas GT"], 2)

    def test_false_and_missing_notes(self):
        reference = make_midi([(0.1, 0.5, 60), (0.6, 1.0, 64)])
        extra = make_midi([(0.1, 0.5, 60), (0.6, 1.0, 64), (1.2, 1.6, 67)])
        metrics = evaluate_prediction(reference, extra, 2)
        self.assertAlmostEqual(metrics["P_no"], 2 / 3)
        self.assertEqual(metrics["R_no"], 1)
        self.assertAlmostEqual(metrics["F_no"], 0.8)
        missing = make_midi([(0.1, 0.5, 60)])
        metrics = evaluate_prediction(reference, missing, 2)
        self.assertEqual(metrics["P"], 1)
        self.assertEqual(metrics["R"], 0.5)
        self.assertAlmostEqual(metrics["F"], 2 / 3)
        self.assertAlmostEqual(metrics["Acc"], 0.5)

    def test_onset_pitch_and_offset_tolerances(self):
        reference = make_midi([(0.1, 0.6, 60)])
        self.assertEqual(evaluate_prediction(reference, make_midi([(0.149, 0.6, 60)]), 1)["F_no"], 1)
        self.assertEqual(evaluate_prediction(reference, make_midi([(0.151, 0.6, 60)]), 1)["F_no"], 0)
        self.assertEqual(evaluate_prediction(reference, make_midi([(0.1, 0.6, 61)]), 1)["F_no"], 0)
        metrics = evaluate_prediction(reference, make_midi([(0.1, 0.8, 60)]), 1)
        self.assertEqual(metrics["F_no"], 1)
        self.assertEqual(metrics["F"], 0)
        self.assertEqual(evaluate_prediction(reference, make_midi([(0.1, 0.69, 60)]), 1)["F"], 1)
        short = make_midi([(0.1, 0.2, 60)])
        self.assertEqual(evaluate_prediction(short, make_midi([(0.1, 0.249, 60)]), 1)["F"], 1)

    def test_empty_prediction_and_empty_reference_are_distinct(self):
        reference = make_midi([(0, 0.5, 60)])
        empty = make_midi()
        metrics = evaluate_prediction(reference, empty, 1)
        self.assertTrue(all(metrics[key] == 0 for key in METRIC_KEYS))
        for prediction in (reference, empty):
            metrics = evaluate_prediction(empty, prediction, 1)
            self.assertEqual(metrics["estado"], "sin referencia evaluable")
            self.assertTrue(all(metrics[key] is None for key in METRIC_KEYS))

    def test_crop_nonzero_start_and_sustained_boundary_notes(self):
        midi = make_midi([(0, 1, 50), (1, 2.5, 60), (2.5, 3.5, 64), (3.5, 5, 67), (4, 5, 70)])
        cropped = crop_midi(midi, 2, 2)
        intervals, pitches = midi_to_intervals_and_pitches(cropped)
        np.testing.assert_allclose(intervals, [(0, 0.5), (0.5, 1.5), (1.5, 2)])
        self.assertEqual(len(pitches), 3)
        self.assertEqual(midi.instruments[0].notes[1].start, 1)
        # Predicciones que se prolongan fuera del audio se recortan de igual forma.
        prediction = make_midi([(0, 0.5, 60), (0.5, 1.5, 64), (1.5, 9, 67)])
        self.assertEqual(evaluate_prediction(cropped, prediction, 2)["F"], 1)

    def test_multiple_reference_files_and_drum_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f"{name}.mid" for name in ("one", "two", "drums")]
            for path, midi in zip(paths, [make_midi([(1, 2.5, 60)]),
                make_midi([(2.5, 3.5, 64)]), make_midi([(2, 3, 36)], is_drum=True)]):
                midi.write(str(path))
            combined = load_reference_midis([*paths, paths[0]], 2, 1)
        intervals, _ = midi_to_intervals_and_pitches(combined)
        np.testing.assert_allclose(intervals, [(0, 0.5), (0.5, 1)])
        self.assertEqual(len(combined.instruments), 2)

    def test_frame_frequencies_are_unique_and_end_is_exclusive(self):
        midi = make_midi([(0, 0.02, 60), (0, 0.02, 60)])
        frames = midi_to_frame_frequencies(midi, 0.03)
        self.assertEqual([len(frame) for frame in frames], [1, 1, 0])
        self.assertEqual(frames[0][0], pretty_midi.note_number_to_hz(60))

    def test_variant_rows_post_dsp_and_deltas(self):
        reference = make_midi([(0.1, 0.5, 60)])
        wrong = make_midi([(0.1, 0.5, 60), (0.6, 1, 64)])
        rows = benchmark_variants({"raw": result(wrong), "filtered": result(wrong, reference)}, reference, 1)
        self.assertEqual([(row["Variante"], row["Salida"]) for row in rows],
                         [("raw", "modelo"), ("filtered", "modelo"), ("filtered", "post-DSP")])
        self.assertEqual(rows[0]["Δ F"], 0)
        self.assertAlmostEqual(rows[2]["Δ F"], 1 / 3)
        without_raw = benchmark_variants({"filtered": result(reference)}, reference, 1)
        self.assertEqual(without_raw[0]["F_no"], 1)
        self.assertIsNone(without_raw[0]["Δ F_no"])
        empty_ref = benchmark_variants({"raw": result(reference)}, make_midi(), 1)
        self.assertIsNone(empty_ref[0]["Δ F"])

    def test_invalid_windows_untranscribed_and_drums(self):
        midi = make_midi()
        for start, duration in [(-1, 1), (0, 0), (0, -1), (np.nan, 1), (0, np.inf)]:
            with self.assertRaises(ValueError):
                crop_midi(midi, start, duration)
        with self.assertRaises(ValueError):
            benchmark_variants({"drums": result(midi, stem="drums")}, midi, 1)
        with self.assertRaises(ValueError):
            benchmark_variants({"raw": SimpleNamespace(stem="bass", transcription=None)}, midi, 1)


if __name__ == "__main__":
    unittest.main()
