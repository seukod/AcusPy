import os
import warnings
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import scipy.io.wavfile
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

warnings.filterwarnings('ignore')

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 22050
AUDIO_PATH = PROJECT_ROOT / "data/babyslakh_16k/Track00001/stems/S03.wav"

print(f"Cargando audio original desde {AUDIO_PATH}...")
audio_signal, sr = librosa.load(str(AUDIO_PATH), sr=TARGET_SR, mono=True)
# Guardar el original resampleado
scipy.io.wavfile.write(str(OUTPUTS_DIR / "1_original_bass.wav"), sr, (audio_signal * 32767).astype(np.int16))

print("Inferencia con Basic Pitch...")
model_output, bp_midi, note_events = predict(str(AUDIO_PATH), model_or_model_path=ICASSP_2022_MODEL_PATH)

print("Aplicando DSP...")
filtered_notes = list(note_events)
MIN_DURATION_SEC = 0.05
filtered_notes = [n for n in filtered_notes if (n[1] - n[0]) >= MIN_DURATION_SEC]
PITCH_MIN, PITCH_MAX = 28, 67
filtered_notes = [n for n in filtered_notes if PITCH_MIN <= n[2] <= PITCH_MAX]

dsp_midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=33, name='Electric Bass (finger)')
for n in filtered_notes:
    vel_127 = int(np.round(127 * n[3])) if n[3] <= 1.0 else int(n[3])
    vel_127 = max(1, min(127, vel_127))
    instrument.notes.append(pretty_midi.Note(velocity=vel_127, pitch=int(n[2]), start=n[0], end=n[1]))
dsp_midi.instruments.append(instrument)

def synthesize_and_save(pm, path, fs=TARGET_SR):
    audio = pm.synthesize(fs=fs)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    scipy.io.wavfile.write(str(path), fs, (audio * 32767).astype(np.int16))

print("Sintetizando audios MIDI...")
synthesize_and_save(bp_midi, OUTPUTS_DIR / "2_basic_pitch_crudo.wav")
synthesize_and_save(dsp_midi, OUTPUTS_DIR / "3_basic_pitch_dsp.wav")

print(f"Audios exportados exitosamente a '{OUTPUTS_DIR}'")
