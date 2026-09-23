# Copyright 2026 AcusPy contributors
#
# BabySlakh/Slakh dataset plumbing for the audio JEPA: track discovery,
# MIDI-derived per-frame targets, a multiprocess CQT cache, and the clip
# dataset that feeds [8, 256, T] HCQTs plus pitch/onset targets.
#
# Designed against BabySlakh (Zenodo 4603870) but works with any layout of
# `<root>/<track>/audio.wav` + `<root>/<track>/**.mid` (Slakh redux).
#
# Run interactively with `marimo edit jepa/babyslakh.py`.

import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    import sys
    from pathlib import Path

    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    import hcqt

    N_PITCH_CLASSES = 88  # MIDI notes 21..108 (A0..C8)
    PITCH_OFFSET = 21


@app.function
def list_tracks(root):
    """Discover BabySlakh/Slakh tracks under `root`.

    Returns a list of dicts: {"name", "audio", "midis"}. Empty if `root`
    does not exist (e.g. dataset not downloaded yet).
    """
    root = Path(root)
    if not root.exists():
        return []
    tracks = []
    for track_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        wav = track_dir / "mix.wav"
        if not wav.exists():
            wav = track_dir / "audio.wav"
        if not wav.exists():
            continue
        midis = sorted(track_dir.rglob("*.mid"))
        if not midis:
            continue
        tracks.append(
            {"name": track_dir.name, "audio": str(wav), "midis": [str(m) for m in midis]}
        )
    return tracks


@app.function
def midi_targets(midis, n_frames, fs=hcqt.FRAMES_PER_SECOND):
    """Frame-level targets from MIDI files.

    Returns (pitch, onset, onset_pitch): pitch is [n_frames, 88]
    active-semitone flags, onset is [n_frames] flags (any instrument),
    onset_pitch is [n_frames, 88] per-semitone onset flags. All aligned to
    CQT frames (fs = SR / FFT_HOP).
    """
    import pretty_midi

    pitch = np.zeros((n_frames, N_PITCH_CLASSES), dtype=np.uint8)
    onset = np.zeros(n_frames, dtype=np.uint8)
    onset_pitch = np.zeros((n_frames, N_PITCH_CLASSES), dtype=np.uint8)
    for path in midis:
        pm = pretty_midi.PrettyMIDI(str(path))
        notes = [note for inst in pm.instruments for note in inst.notes]
        for note in notes:
            lo = max(int(round(note.start * fs)), 0)
            hi = min(int(np.ceil(note.end * fs)), n_frames)
            if lo < hi and PITCH_OFFSET <= note.pitch <= PITCH_OFFSET + N_PITCH_CLASSES - 1:
                pitch[lo:hi, note.pitch - PITCH_OFFSET] = 1
                on = int(round(note.start * fs))
                if 0 <= on < n_frames:
                    onset[on] = 1
                    onset_pitch[on, note.pitch - PITCH_OFFSET] = 1
    return pitch, onset, onset_pitch


@app.function
def _process_track(job):
    """Worker: magnitude CQT + MIDI targets for one track -> cache files."""
    name, audio_path, midis, cache_dir, fs = job
    import soundfile as sf

    cache_dir = Path(cache_dir)
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    magnitude = hcqt.cqt_magnitude(audio, sr)  # [297, T] raw magnitudes
    n_frames = magnitude.shape[1]
    pitch, onset, onset_pitch = midi_targets(midis, n_frames)
    np.save(cache_dir / f"{name}_cqt.npy", magnitude.astype(np.float16))
    np.save(cache_dir / f"{name}_pitch.npy", pitch)
    np.save(cache_dir / f"{name}_onset.npy", onset)
    np.save(cache_dir / f"{name}_onset_pitch.npy", onset_pitch)
    return name, n_frames


@app.function
def ensure_cache(root, cache_dir, workers=8):
    """Precompute the per-track CQT cache (idempotent, multiprocess).

    Returns the list of cached track names. Skips tracks already cached.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tracks = list_tracks(root)
    if not tracks:
        raise FileNotFoundError(
            f"No tracks found under {root}. Download BabySlakh from "
            "https://zenodo.org/records/4603870 and extract it there."
        )
    jobs = [
        (t["name"], t["audio"], t["midis"], str(cache_dir), hcqt.FRAMES_PER_SECOND)
        for t in tracks
        if not (cache_dir / f"{t['name']}_cqt.npy").exists()
    ]
    if jobs:
        print(f"Caching {len(jobs)}/{len(tracks)} tracks with {workers} workers...")
        if workers <= 1:
            for job in jobs:
                _process_track(job)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_process_track, job) for job in jobs]
                for i, future in enumerate(as_completed(futures)):
                    name, frames = future.result()
                    if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                        print(f"  [{i + 1}/{len(jobs)}] {name} ({frames} frames)")
    return sorted(p.name[: -len("_cqt.npy")] for p in cache_dir.glob("*_cqt.npy"))


@app.function
class BabySlakhClips(torch.utils.data.Dataset):
    """Random HCQT clips with pitch/onset targets from the cache.

    Splits by track (no leakage): a deterministic fraction of tracks goes
    to validation. Returns dicts with:
      hcqt   [8, 256, frames]  per-crop normalized log
      pitch  [frames, 88]      active semitone flags
      onset  [frames]          onset flags
    """

    def __init__(self, cache_dir, frames=96, split="train", val_frac=0.2, seed=0, crops_per_track=64):
        self.cache_dir = Path(cache_dir)
        self.frames = frames
        names = sorted(p.name[: -len("_cqt.npy")] for p in self.cache_dir.glob("*_cqt.npy"))
        if not names:
            raise FileNotFoundError(f"empty cache at {self.cache_dir}; run ensure_cache first")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(names))
        n_val = int(len(names) * val_frac) if val_frac > 0 else 0
        picked = order[:n_val] if split == "val" else order[n_val:]
        self.names = [names[i] for i in picked]
        self.crops_per_track = crops_per_track
        self.seed = seed

    def __len__(self):
        return len(self.names) * self.crops_per_track

    def __getitem__(self, index):
        name = self.names[index % len(self.names)]
        cqt = np.load(self.cache_dir / f"{name}_cqt.npy", mmap_mode="r")
        total = cqt.shape[1]
        rng = np.random.default_rng(self.seed + index)
        start = int(rng.integers(0, max(total - self.frames, 1)))
        sl = slice(start, start + self.frames)
        magnitude = np.asarray(cqt[:, sl], dtype=np.float32)
        hcqt_clip = (
            hcqt.harmonic_stacking(hcqt.normalized_log(torch.from_numpy(magnitude)))
            .numpy()
            .astype(np.float32)
        )  # [8, 256, frames]
        pitch = np.load(self.cache_dir / f"{name}_pitch.npy", mmap_mode="r")[sl]
        onset = np.load(self.cache_dir / f"{name}_onset.npy", mmap_mode="r")[sl]
        onset_pitch = np.load(self.cache_dir / f"{name}_onset_pitch.npy", mmap_mode="r")[sl]
        return {
            "hcqt": torch.from_numpy(hcqt_clip),
            "pitch": torch.from_numpy(np.asarray(pitch, dtype=np.float32)),
            "onset": torch.from_numpy(np.asarray(onset, dtype=np.float32)),
            "onset_pitch": torch.from_numpy(np.asarray(onset_pitch, dtype=np.float32)),
        }


@app.function
class SyntheticClips(torch.utils.data.Dataset):
    """Smoke-test dataset with the same shapes (no real audio needed)."""

    def __init__(self, size=512, frames=96, seed=0):
        self.size, self.frames, self.seed = size, frames, seed

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(self.seed + index)
        hcqt_clip = torch.rand(8, 256, self.frames, generator=g)
        pitch = (torch.rand(self.frames, 88, generator=g) < 0.02).float()
        onset = (torch.rand(self.frames, generator=g) < 0.02).float()
        onset_pitch = (torch.rand(self.frames, 88, generator=g) < 0.02).float()
        return {"hcqt": hcqt_clip, "pitch": pitch, "onset": onset, "onset_pitch": onset_pitch}


@app.cell
def _():
    """Doc cell."""
    import marimo as mo

    mo.md(
        """
        # BabySlakh cache & clips

        1. Descarga BabySlakh (Zenodo 4603870) y extraelo en `babyslakh_16k/`.
        2. Desde `train_audio.py`: `ensure_cache(...)` precomputa el CQT por track.
        3. `BabySlakhClips` entrega clips de HCQT + targets de pitch/onset.
        """
    )
    return (mo,)


@app.cell
def _():
    """Shape smoke test for both datasets."""
    import tempfile

    synth = SyntheticClips(size=8)
    item = synth[0]
    assert item["hcqt"].shape == (8, 256, 96)
    assert item["pitch"].shape == (96, 88)
    assert item["onset"].shape == (96,)
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        for name in ("T1", "T2"):
            np.save(Path(tmp) / f"{name}_cqt.npy", rng.random((297, 300)).astype(np.float16))
            np.save(Path(tmp) / f"{name}_pitch.npy", rng.integers(0, 2, (300, 88)).astype(np.uint8))
            np.save(Path(tmp) / f"{name}_onset.npy", rng.integers(0, 2, 300).astype(np.uint8))
            np.save(Path(tmp) / f"{name}_onset_pitch.npy", rng.integers(0, 2, (300, 88)).astype(np.uint8))
        ds = BabySlakhClips(tmp, frames=96, val_frac=0, crops_per_track=1)
        assert len(ds) == 2
        item = ds[0]
        assert item["hcqt"].shape == (8, 256, 96), item["hcqt"].shape
        assert item["pitch"].shape == (96, 88)
    print("dataset shapes OK")
    return


if __name__ == "__main__":
    app.run()
