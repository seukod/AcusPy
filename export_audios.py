import os
import numpy as np
import librosa
import pretty_midi
import scipy.io.wavfile
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

TARGET_SR = 22050
AUDIO_PATH = "data/babyslakh_16k/Track00001/stems/S03.wav"

print("Cargando audio original...")
audio_signal, sr = librosa.load(AUDIO_PATH, sr=TARGET_SR, mono=True)
# Guardar el original resampleado
scipy.io.wavfile.write("outputs/1_original_bass.wav", sr, (audio_signal * 32767).astype(np.int16))

print("Inferencia con Basic Pitch...")
model_output, bp_midi, note_events = predict(AUDIO_PATH, model_or_model_path=ICASSP_2022_MODEL_PATH)

print("Aplicando DSP...")
filtered_notes = list(note_events)
MIN_DURATION_SEC = 0.05
filtered_notes = [n for n in filtered_notes if (n[1] - n[0]) >= MIN_DURATION_SEC]
PITCH_MIN, PITCH_MAX = 28, 67
filtered_notes = [n for n in filtered_notes if PITCH_MIN <= n[2] <= PITCH_MAX]

dsp_midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=33, name='Electric Bass (finger)')
for n in filtered_notes:
    instrument.notes.append(pretty_midi.Note(velocity=int(n[3]), pitch=int(n[2]), start=n[0], end=n[1]))
dsp_midi.instruments.append(instrument)

def synthesize_and_save(pm, path, fs=TARGET_SR):
    audio = pm.synthesize(fs=fs)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    scipy.io.wavfile.write(path, fs, (audio * 32767).astype(np.int16))

print("Sintetizando audios MIDI...")
synthesize_and_save(bp_midi, "outputs/2_basic_pitch_crudo.wav")
synthesize_and_save(dsp_midi, "outputs/3_basic_pitch_dsp.wav")

print("Audios exportados exitosamente a la carpeta 'outputs/'")
