"""Pruebas de orquestación sin descargas de pesos ni inferencia costosa.

Ejecutar: .venv/bin/python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pretty_midi
import soundfile as sf
import torch

from src import (
    AudioArtifact, AudioPreProcessor, BasicPitchTranscriber, DemucsTranscriber,
    DSPVariant, NotePostProcessor, SeparationResult,
    available_devices,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.sin(2 * np.pi * 220 * np.arange(1600) / 16000).astype(np.float32) * 0.2
        self.engine = DemucsTranscriber()
        self.model = SimpleNamespace(samplerate=44100, audio_channels=2,
                                     sources=["drums", "bass", "other", "vocals"])
        self.engine._separator_model = self.model

    def test_device_detection_and_explicit_mps_selection(self):
        with patch("torch.backends.mps.is_available", return_value=True), patch(
            "torch.cuda.is_available", return_value=False
        ):
            self.assertEqual(available_devices(), ["cpu", "mps"])
            engine = DemucsTranscriber(device="mps")
            self.assertEqual(engine.device, "mps")
            self.assertIsNone(engine._separator_model)

    def test_cpu_only_and_unavailable_accelerators(self):
        with patch("torch.backends.mps.is_available", return_value=False), patch(
            "torch.cuda.is_available", return_value=False
        ):
            self.assertEqual(available_devices(), ["cpu"])
            for device in ("mps", "cuda", "invalid"):
                with self.assertRaisesRegex(ValueError, "no está disponible"):
                    DemucsTranscriber(device=device)

    def test_cuda_detection(self):
        with patch("torch.backends.mps.is_available", return_value=False), patch(
            "torch.cuda.is_available", return_value=True
        ):
            self.assertEqual(available_devices(), ["cpu", "cuda"])
            self.assertEqual(DemucsTranscriber(device="cuda").device, "cuda")

    def separate(self, audio=None):
        def apply(model, mix, **kwargs):
            self.assertEqual(tuple(mix.shape), (1, 2, 4410))
            self.assertFalse(torch.is_grad_enabled())
            return mix[:, None].repeat(1, 4, 1, 1)

        with patch("demucs.apply.apply_model", side_effect=apply) as mocked:
            result = self.engine.separate(self.audio if audio is None else audio, sr=16000)
            self.assertEqual(mocked.call_count, 1)
        return result

    def test_resampling_mono_stereo_and_duration(self):
        result = self.separate()
        self.assertEqual(result.input_audio.sr, 16000)
        self.assertEqual(result.demucs_input.audio.shape, (2, 4410))
        self.assertEqual(set(result.stems), set(self.model.sources))
        np.testing.assert_allclose(result.stems["bass"].audio,
                                   result.demucs_input.audio, atol=1e-6)
        self.assertAlmostEqual(result.stems["bass"].duration, 0.1)
        stereo = np.stack([self.audio, self.audio * 0.25])
        separated = self.separate(stereo)
        self.assertFalse(np.allclose(separated.demucs_input.audio[0], separated.demucs_input.audio[1]))

    def test_silence_is_finite(self):
        result = self.separate(np.zeros_like(self.audio))
        self.assertTrue(np.isfinite(result.stems["vocals"].audio).all())
        np.testing.assert_array_equal(result.stems["vocals"].audio, 0)

    def test_file_input_keeps_native_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            sf.write(path, self.audio, 16000, subtype="FLOAT")
            with patch("demucs.apply.apply_model", side_effect=lambda model, mix, **kw: mix[:, None].repeat(1, 4, 1, 1)):
                result = self.engine.separate(path)
        self.assertEqual(result.input_audio.sr, 16000)

    def test_model_is_loaded_once(self):
        engine = DemucsTranscriber()
        with patch("demucs.pretrained.get_model") as load:
            self.assertIs(engine.separator_model, engine.separator_model)
        load.assert_called_once_with("htdemucs")

    def test_independent_branches_and_stage_snapshots(self):
        separated = self.separate()
        def in_place(audio, sr):
            audio *= 0.5
            return audio
        half = AudioPreProcessor(name="half", custom_transforms=[in_place])
        variants = {"bass": [DSPVariant("chain", [half, AudioPreProcessor(
            name="quarter", custom_transforms=[in_place])]), DSPVariant("raw")]}
        result = self.engine.process_variants(separated, variants)["bass"]
        raw = result["raw"].final_audio
        self.assertEqual(raw.sr, 22050)
        self.assertEqual(raw.audio.shape, (2205,))
        np.testing.assert_allclose(result["chain"].stages["half"].audio, raw.audio * 0.5)
        np.testing.assert_allclose(result["chain"].final_audio.audio, raw.audio * 0.25)
        self.assertFalse(np.shares_memory(raw.audio, result["chain"].stages["input_22050"].audio))

    def test_transcription_is_explicit_and_does_not_mutate_dsp_results(self):
        dsp = self.engine.process_variants(self.separate(), {"bass": [DSPVariant("raw")]})
        self.engine.transcriber = Mock()
        self.engine.transcriber.transcribe.return_value = {"midi": "sentinel"}
        self.engine.transcriber.transcribe.assert_not_called()
        result = self.engine.transcribe_variants(dsp)
        self.assertIsNone(dsp["bass"]["raw"].transcription)
        self.assertEqual(result["bass"]["raw"].transcription["midi"], "sentinel")
        self.assertEqual(result["bass"]["raw"].transcription["pipeline_name"], "raw")
        self.assertFalse(result["bass"]["raw"].transcription["applied_pre_dsp"])
        self.assertEqual(self.engine.transcriber.transcribe.call_args.kwargs["sr"], 22050)

    def test_export_stereo_float_intermediates_and_midis(self):
        separated = self.separate()
        midi = pretty_midi.PrettyMIDI()
        self.engine.transcriber = Mock()
        self.engine.transcriber.transcribe.return_value = {
            "midi": midi, "raw_midi": midi, "note_events": [], "raw_note_events": [],
        }
        dsp = self.engine.process_variants(separated, {"bass": [DSPVariant("raw")]})
        result = self.engine.transcribe_variants(dsp)
        with tempfile.TemporaryDirectory() as directory:
            root = self.engine.export(separated, result, directory)
            info = sf.info(root / "stems/bass.wav")
            self.assertEqual((info.channels, info.samplerate, info.subtype), (2, 44100, "FLOAT"))
            self.assertEqual(sf.info(root / "variants/bass/raw/00_input_22050.wav").samplerate, 22050)
            self.assertTrue((root / "variants/bass/raw/midi.mid").exists())
            self.assertTrue((root / "variants/bass/raw/raw_midi.mid").exists())
            self.assertTrue((root / "manifest.json").exists())

    def test_invalid_inputs_and_names(self):
        for audio in [np.array([]), np.array([np.nan]), np.ones((10, 2))]:
            with self.assertRaises(ValueError):
                AudioArtifact(audio, 16000)
        with self.assertRaises(ValueError):
            self.engine.separate(self.audio)
        with self.assertRaises(ValueError):
            DSPVariant("../escape")
        with self.assertRaises(ValueError):
            DSPVariant("same", [AudioPreProcessor(name="x"), AudioPreProcessor(name="x")])
        separated = self.separate()
        with self.assertRaises(ValueError):
            self.engine.process_variants(separated, {"guitar": [DSPVariant("raw")]})
        with self.assertRaises(ValueError):
            self.engine.process_variants(separated, {"bass": [DSPVariant("raw"), DSPVariant("raw")]})

    def test_stage_must_preserve_duration(self):
        stage = AudioPreProcessor(name="bad", custom_transforms=[lambda audio, sr: audio[:-1]])
        with self.assertRaisesRegex(ValueError, "conservar"):
            self.engine.process_variants(self.separate(), {"bass": [DSPVariant("bad", [stage])]})


class BasicPitchTests(unittest.TestCase):
    def test_model_reuse_float_audio_and_cleanup(self):
        captured_paths = []
        midi = pretty_midi.PrettyMIDI()
        events = [(0.0, 0.01, 60, 0.8, None), (0.0, 0.1, 64, 0.8, None)]
        def inference(path, model_or_model_path):
            captured_paths.append(Path(path))
            audio, sr = sf.read(path)
            self.assertEqual(sr, 22050)
            self.assertGreater(float(audio.max()), 1.0)  # No clipping PCM16.
            return {}, midi, events
        with patch("src.transcriber.Model.__init__", return_value=None) as load, patch(
            "src.transcriber.predict", side_effect=inference
        ):
            transcriber = BasicPitchTranscriber("model.onnx")
            load.assert_not_called()
            for _ in range(2):
                result = transcriber.transcribe(np.ones(2205, dtype=np.float32) * 1.2,
                    note_postprocessor=NotePostProcessor(min_duration_sec=0.05))
            load.assert_called_once_with("model.onnx")
            self.assertEqual(len(result["note_events"]), 1)
            self.assertEqual(len(result["raw_note_events"]), 2)
        self.assertTrue(all(not path.exists() for path in captured_paths))

    def test_temporary_file_removed_on_inference_failure(self):
        paths = []
        def fail(path, **kwargs):
            paths.append(Path(path))
            raise RuntimeError("inference failure")
        transcriber = BasicPitchTranscriber()
        transcriber._model = object()
        with patch("src.transcriber.predict", side_effect=fail):
            with self.assertRaises(RuntimeError):
                transcriber.transcribe(np.ones(100, dtype=np.float32))
        self.assertTrue(all(not path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
