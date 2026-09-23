# Copyright 2026 AcusPy contributors
#
# Data augmentation by remixing BabySlakh stems: each remix mixes 2-5
# non-drum stems from random training tracks with random gains, and writes a
# BabySlakh-compatible track dir (`mix.wav` + merged MIDI) so that
# `babyslakh.list_tracks` / `ensure_cache` consume it unchanged.
#
# The ground-truth MIDI of a remix is exactly the union of the stems' notes
# at their original positions, so labels stay perfect.
#
# Run:  python jepa/remix.py --n-remixes 400

import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    import argparse
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    import babyslakh

    def pad_add(a, b):
        n = max(len(a), len(b))
        out = np.zeros(n, dtype=np.float32)
        out[: len(a)] += a
        out[: len(b)] += b
        return out


@app.cell
def _():
    """CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="babyslakh_16k")
    parser.add_argument("--output", default="babyslakh_remix")
    parser.add_argument("--n-remixes", type=int, default=400)
    parser.add_argument("--min-stems", type=int, default=2)
    parser.add_argument("--max-stems", type=int, default=5)
    parser.add_argument("--min-gain-db", type=float, default=-9.0)
    parser.add_argument("--max-gain-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-tracks", nargs="*", default=[], help="Val track names")
    args = parser.parse_args([] if mo.running_in_notebook() else None)
    return (args,)


@app.cell
def _():
    """Stem inventory: non-drum stems per track from metadata.yaml."""
    import yaml

    def track_stems(track_dir):
        """{stem_id: {"inst_class", "is_drum"}} for a track dir."""
        meta_path = Path(track_dir) / "metadata.yaml"
        if not meta_path.exists():
            return {}
        meta = yaml.safe_load(meta_path.read_text()) or {}
        stems = {}
        for stem_id, info in (meta.get("stems") or {}).items():
            if not info.get("is_drum", False):
                stems[stem_id] = {
                    "inst_class": info.get("inst_class", "?"),
                    "midi_program": info.get("program_num", 0),
                }
        return stems

    return (track_stems,)


@app.cell
def _(args, track_stems):
    """Build the remix plan (which stems, which gains) and write track dirs."""
    import soundfile as sf

    root = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = []  # (track_dir, stem_id, audio_path, midi_path, info)
    for track in babyslakh.list_tracks(args.data):
        if track["name"] in args.exclude_tracks:
            continue
        stems = track_stems(track["audio"].rsplit("/", 1)[0])
        for stem_id, info in stems.items():
            wav = Path(track["audio"]).parent / "stems" / f"{stem_id}.wav"
            mid = Path(track["audio"]).parent / "MIDI" / f"{stem_id}.mid"
            if wav.exists() and mid.exists():
                pool.append((track["name"], stem_id, str(wav), str(mid), info))
    print(f"stem pool: {len(pool)} stems from {len(set(p[0] for p in pool))} tracks")

    rng = np.random.default_rng(args.seed)
    manifest = []
    n = len(str(args.n_remixes))
    for i in range(args.n_remixes):
        k = int(rng.integers(args.min_stems, args.max_stems + 1))
        picks = [pool[j] for j in rng.choice(len(pool), size=k, replace=False)]
        gains = 10.0 ** (rng.uniform(args.min_gain_db, args.max_gain_db, size=k) / 20.0)

        remix_dir = out_dir / f"REMIX_{str(i).zfill(4)}"
        remix_dir.mkdir(parents=True, exist_ok=True)
        (remix_dir / "stems").mkdir(exist_ok=True)
        (remix_dir / "MIDI").mkdir(exist_ok=True)

        import pretty_midi as pm

        merged = pm.PrettyMIDI()
        mix = None
        sr = None
        for (track_name, stem_id, wav, mid, info), gain in zip(picks, gains):
            audio, sr = sf.read(wav, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1) * gain
            mix = audio if mix is None else pad_add(mix, audio)
            stem_out = remix_dir / "stems" / f"{track_name}_{stem_id}.wav"
            sf.write(stem_out, audio, sr)
            midi = pm.PrettyMIDI(mid)
            for inst in midi.instruments:
                merged.instruments.append(inst)
            stem_mid = remix_dir / "MIDI" / f"{track_name}_{stem_id}.mid"
            midi.write(str(stem_mid))
            manifest.append(
                {
                    "remix": remix_dir.name,
                    "track": track_name,
                    "stem": stem_id,
                    "gain_db": float(20 * np.log10(gain)),
                    "inst_class": info["inst_class"],
                }
            )
        peak = float(np.abs(mix).max())
        if peak > 0.99:
            mix = mix * (0.99 / peak)
        sf.write(remix_dir / "mix.wav", mix, sr)
        merged.write(str(remix_dir / "all_src.mid"))
        if (i + 1) % 50 == 0:
            print(f"  wrote {i + 1}/{args.n_remixes} remixes")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {args.n_remixes} remixes in {out_dir} (manifest.json written)")
    return (out_dir,)


@app.cell
def _(out_dir):
    """Preview: verify the remixes are discoverable as normal tracks."""
    tracks = babyslakh.list_tracks(str(out_dir))
    print(f"discoverable tracks: {len(tracks)}")
    if tracks:
        print("example:", tracks[0]["name"], "->", tracks[0]["audio"])
    return (tracks,)


if __name__ == "__main__":
    app.run()
