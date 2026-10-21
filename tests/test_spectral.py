"""Comprueba escalas y referencias espectrales compartidas."""
import unittest

import numpy as np

from src import AudioArtifact
from src.spectral import spectral_comparison


class SpectralTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(22050, dtype=np.float32) / 22050
        self.original = AudioArtifact(np.sin(2 * np.pi * 440 * time) * 0.2, 22050)
        self.half = AudioArtifact(self.original.audio * 0.5, 22050)

    def test_attenuation_is_preserved_for_all_views(self):
        for representation in ("STFT", "Mel", "log-Mel"):
            with self.subTest(view=representation):
                result = spectral_comparison({"original": self.original, "half": self.half},
                                             self.original, representation)
                first, half = result.matrices.values()
                peak = np.unravel_index(np.argmax(first), first.shape)
                if representation == "Mel":
                    self.assertAlmostEqual(half[peak] / first[peak], 0.25, places=5)
                else:
                    self.assertAlmostEqual(float(half[peak] - first[peak]), -6.0206, places=3)

    def test_hiding_raw_does_not_change_reference_or_matrix(self):
        for representation in ("STFT", "Mel", "log-Mel"):
            together = spectral_comparison({"raw": self.original, "half": self.half}, self.original, representation)
            alone = spectral_comparison({"half": self.half}, self.original, representation)
            self.assertEqual(together.reference, alone.reference)
            np.testing.assert_array_equal(together.matrices["half"], alone.matrices["half"])
            self.assertEqual((together.vmin, together.vmax), (alone.vmin, alone.vmax))
            if representation != "STFT":
                self.assertEqual(alone.matrices["half"].shape[0], 128)

    def test_silence_is_finite_and_has_stable_color_limits(self):
        silence = AudioArtifact(np.zeros(22050, dtype=np.float32), 22050)
        for representation in ("STFT", "Mel", "log-Mel"):
            result = spectral_comparison({"silent": silence}, silence, representation)
            self.assertTrue(np.isfinite(result.matrices["silent"]).all())
            self.assertTrue(np.isfinite([result.reference, result.vmin, result.vmax]).all())
            self.assertGreater(result.vmax, result.vmin)
            self.assertLessEqual(result.matrices["silent"].max(), result.vmin)

    def test_input_rates_and_stereo_are_converted(self):
        reference = AudioArtifact(np.ones((2, 16000), dtype=np.float32) * 0.1, 16000)
        result = spectral_comparison({"original": reference}, reference)
        self.assertEqual(result.matrices["original"].shape, (128, 87))

    def test_mismatched_fragments_and_invalid_representation(self):
        with self.assertRaises(ValueError):
            spectral_comparison({"short": AudioArtifact(np.ones(1000), 22050)}, self.original)
        with self.assertRaises(ValueError):
            spectral_comparison({"original": self.original}, self.original, "invalid")


if __name__ == "__main__":
    unittest.main()
