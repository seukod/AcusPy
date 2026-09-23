# Copyright 2026 AcusPy contributors
#
# Transcription evaluation on BabySlakh validation tracks, comparing:
#   A) Spotify's pretrained basic-pitch (official icassp_2022 checkpoint)
#   B) our supervised NMP (same architecture, trained from scratch on
#      BabySlakh with identical data as the JEPA)
#   C) our JEPA (AudioLeWM) with a frozen encoder + light heads
#
# Metrics follow the paper: note F-measure without offset (Fno, main),
# note F-measure (F) and frame accuracy, via mir_eval. Reports mean and
# std across tracks so the variance between models is visible.
#
# Requires trained checkpoints from train_audio.py and train_nmp.py.

import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    import argparse
    import sys
    import tempfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pretty_midi
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).parent))
    import babyslakh
    import hcqt
    import nmp_encoder
    import audio_lewm


@app.cell
def _():
    """CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="babyslakh_16k")
    parser.add_argument("--cache", default=".cache/babyslakh_hcqt")
    parser.add_argument("--jepa-checkpoint", default=None, help="lewm_audio.pt")
    parser.add_argument("--nmp-checkpoint", default=None, help="nmp.pt")
    parser.add_argument("--n-val-tracks", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=0, help="Evaluate only the first N seconds (0 = full track)")
    parser.add_argument("--head-epochs", type=int, default=10, help="Epochs for JEPA head training")
    parser.add_argument("--head-batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remix-cache", default=None, help="Optional remix CQT cache for head training")
    parser.add_argument("--remix-crops", type=int, default=8)
    args = parser.parse_args([] if mo.running_in_notebook() else None)
    return (args,)


@app.cell
def _(args):
    """Validation tracks (same deterministic split as training)."""
    val_ds = babyslakh.BabySlakhClips(args.cache, args.frames, "val", 0.2, args.seed)
    tracks = [t for t in babyslakh.list_tracks(args.data) if t["name"] in val_ds.names]
    tracks = tracks[: args.n_val_tracks]
    print(f"evaluating on {len(tracks)} val tracks: {[t['name'] for t in tracks]}")
    return (tracks,)


@app.cell
def _():
    """Shared helpers: ground truth, HCQT, posteriorgrams -> notes, metrics."""

    def ground_truth_notes(track):
        """mir_eval (intervals [n,2], pitches [n]) from the track's MIDI.

        Prefers all_src.mid (the complete mixture annotation) when present.
        """
        chosen = [p for p in track["midis"] if p.endswith("all_src.mid")] or track["midis"]
        pm = pretty_midi.PrettyMIDI(chosen[0])
        notes = []
        for inst in pm.instruments:
            for note in inst.notes:
                if 21 <= note.pitch <= 127:
                    notes.append((note.start, note.end, note.pitch))
        notes.sort()
        if not notes:
            return np.zeros((0, 2)), np.zeros(0, dtype=int)
        intervals = np.array([(s, e) for s, e, _ in notes])
        pitches = np.array([p for _, _, p in notes], dtype=int)
        return intervals, pitches

    def track_hcqt(track, cache_dir, seconds=0):
        """Per-crop normalized HCQT [1, 8, 256, T] from the CQT cache."""
        mag = np.load(Path(cache_dir) / f"{track['name']}_cqt.npy", mmap_mode="r")
        total = mag.shape[1]
        n = total if seconds <= 0 else min(total, int(seconds * hcqt.FRAMES_PER_SECOND))
        magnitude = np.asarray(mag[:, :n], dtype=np.float32)
        return hcqt.harmonic_stacking(hcqt.normalized_log(torch.from_numpy(magnitude)))[
            None
        ].float()

    @torch.no_grad()
    def chunked_posteriorgrams(model, h, window=512, overlap=32):
        """Run a conv model over a long track in overlapping chunks.

        Returns dict of posteriorgram arrays with shape [T, bins].
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        t = h.shape[-1]
        outs = []
        start = 0
        while start < t:
            end = min(start + window, t)
            chunk = h[:, :, :, start:end].to(device)
            out = model(chunk)
            logits = {k: v.float().sigmoid().cpu() for k, v in out.items()}
            if start == 0:
                outs.append(logits)
            else:
                outs.append({k: v[:, :, overlap:] for k, v in logits.items()})
            if end == t:
                break
            start = end - overlap
        keys = outs[0].keys()
        return {k: torch.cat([o[k] for o in outs], dim=-1)[0].numpy().T for k in keys}

    def posteriorgrams_to_notes(frames, onset_thresh=0.5, frame_thresh=0.3):
        """basic-pitch post-processing: dict of [T, bins] arrays -> mir_eval notes.

        Our note/onset heads emit 86 semitone bins (C1..E8); basic-pitch's
        post-processing expects 88 bins, so we zero-pad on the high end.
        """
        from basic_pitch.note_creation import model_output_to_notes

        def pad88(x):
            if x.shape[1] < 88:
                x = np.pad(x, ((0, 0), (0, 88 - x.shape[1])))
            return x

        _, note_events = model_output_to_notes(
            {
                "note": pad88(frames["note"]),
                "onset": pad88(frames["onset"]),
                "contour": frames.get("contour", np.zeros_like(frames["note"])),
            },
            onset_thresh=onset_thresh,
            frame_thresh=frame_thresh,
            min_note_len=11,
            include_pitch_bends=False,
        )
        if not note_events:
            return np.zeros((0, 2)), np.zeros(0, dtype=int)
        intervals = np.array([(e[0], e[1]) for e in note_events])
        pitches = np.array([e[2] for e in note_events], dtype=int)
        return intervals, pitches

    def frame_accuracy(ref_int, ref_pitch, est_int, est_pitch, hop=0.01):
        """MIREX-style frame accuracy: fraction of 10ms frames whose active
        pitch sets match exactly between reference and estimate."""

        def active_at(intervals, pitches, times):
            sets = [[] for _ in times]
            for (s, e), p in zip(intervals, pitches):
                lo = int(np.ceil(s / hop))
                hi = min(int(e / hop), len(times) - 1)
                for i in range(max(lo, 0), hi + 1):
                    sets[i].append(p)
            return [frozenset(s) for s in sets]

        t_end = max(
            float(ref_int[:, 1].max()) if len(ref_int) else 0,
            float(est_int[:, 1].max()) if len(est_int) else 0,
        )
        if t_end <= 0:
            return 1.0
        times = np.arange(0, t_end, hop)
        ref_sets = active_at(ref_int, ref_pitch, times)
        est_sets = active_at(est_int, est_pitch, times)
        return np.mean([r == e for r, e in zip(ref_sets, est_sets)])

    def f_scores(ref_int, ref_pitch, est_int, est_pitch):
        """(Fno, F, frame-accuracy) per the paper's metrics."""
        import mir_eval

        if len(ref_int) == 0 and len(est_int) == 0:
            return 1.0, 1.0, 1.0
        if len(ref_int) == 0 or len(est_int) == 0:
            return 0.0, 0.0, 0.0
        fno, _, _, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_int, ref_pitch, est_int, est_pitch, offset_ratio=None
        )
        f, _, _, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_int, ref_pitch, est_int, est_pitch
        )
        acc = frame_accuracy(ref_int, ref_pitch, est_int, est_pitch)
        return float(fno), float(f), float(acc)

    return (ground_truth_notes, track_hcqt, chunked_posteriorgrams, posteriorgrams_to_notes, f_scores)


@app.cell
def _(args):
    """Model A: Spotify's pretrained basic-pitch (official checkpoint)."""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    def basic_pitch_notes(track, seconds=0):
        import soundfile as sf

        audio, sr = sf.read(track["audio"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if seconds > 0:
            audio = audio[: int(seconds * sr)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            _, _, note_events = predict(
                f.name,
                ICASSP_2022_MODEL_PATH,
                onset_threshold=0.5,
                frame_threshold=0.3,
                minimum_note_length=127.7,
            )
        if not note_events:
            return np.zeros((0, 2)), np.zeros(0, dtype=int)
        intervals = np.array([(e[0], e[1]) for e in note_events])
        pitches = np.array([e[2] for e in note_events], dtype=int)
        return intervals, pitches

    return (basic_pitch_notes,)


@app.cell
def _(args):
    """Model B: our supervised NMP checkpoint (from train_nmp.py)."""
    if args.nmp_checkpoint is None:
        print("no NMP checkpoint given; skipping model B")
        nmp_model = None
    else:
        ckpt = torch.load(args.nmp_checkpoint, map_location="cpu", weights_only=False)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        nmp_model = nmp_encoder.NMP(width_scale=ckpt["config"]["width_scale"])
        nmp_model.load_state_dict(state)
    return (nmp_model,)


@app.cell
def _(args):
    """Model C: JEPA encoder (frozen), loaded from lewm_audio.pt."""
    jepa_ckpt = torch.load(args.jepa_checkpoint, map_location="cpu", weights_only=False)
    jepa = audio_lewm.AudioLeWM(
        embed_dim=jepa_ckpt["config"]["latent_dim"],
        width_scale=jepa_ckpt["config"]["width_scale"],
    )
    jepa.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in jepa_ckpt["model"].items()})
    encoder = jepa.encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return (encoder,)


@app.cell
def _(args, encoder):
    """Train the JEPA heads (note + onset) on frozen latents."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_parts = [babyslakh.BabySlakhClips(args.cache, args.frames, "train", 0.2, args.seed)]
    if args.remix_cache:
        train_parts.append(
            babyslakh.BabySlakhClips(
                args.remix_cache, args.frames, "train", 0.0, args.seed, crops_per_track=args.remix_crops
            )
        )
    train_ds = torch.utils.data.ConcatDataset(train_parts)
    loader = DataLoader(
        train_ds,
        batch_size=args.head_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    head_note = nn.Linear(encoder.embed_dim, 86).to(device)
    head_onset = nn.Linear(encoder.embed_dim, 86).to(device)
    opt = torch.optim.AdamW(
        list(head_note.parameters()) + list(head_onset.parameters()), lr=args.head_lr
    )
    pos_weight = torch.tensor(19.0, device=device)
    frozen = encoder.to(device)
    for epoch in range(args.head_epochs):
        total, seen = 0.0, 0
        for batch in loader:
            with torch.no_grad():
                z_lat = frozen(batch["hcqt"].to(device))  # [B, T, D]
            note_logits = head_note(z_lat)
            onset_logits = head_onset(z_lat)
            note_t = batch["pitch"][..., :86].to(device)
            onset_t = batch["onset_pitch"][..., :86].to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                note_logits, note_t
            ) + torch.nn.functional.binary_cross_entropy_with_logits(
                onset_logits, onset_t, pos_weight=pos_weight
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * z_lat.shape[0]
            seen += z_lat.shape[0]
        print(f"head epoch {epoch + 1}/{args.head_epochs} loss={total / max(seen, 1):.4f}")
    return (head_note, head_onset)


@app.cell
def _(
    args,
    basic_pitch_notes,
    chunked_posteriorgrams,
    encoder,
    f_scores,
    ground_truth_notes,
    head_note,
    head_onset,
    nmp_model,
    posteriorgrams_to_notes,
    tracks,
    track_hcqt,
):
    """Run all models on all validation tracks and aggregate."""
    models = {
        "basic-pitch (Spotify)": {},
        "NMP supervised (ours)": {},
        "JEPA + heads (ours)": {},
    }
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    for track in tracks:
        ref_int, ref_pitch = ground_truth_notes(track)
        secs = args.seconds

        est = basic_pitch_notes(track, secs)
        models["basic-pitch (Spotify)"][track["name"]] = f_scores(ref_int, ref_pitch, *est)

        if nmp_model is not None:
            h = track_hcqt(track, args.cache, secs)
            frames = chunked_posteriorgrams(nmp_model, h)
            est = posteriorgrams_to_notes(frames)
            models["NMP supervised (ours)"][track["name"]] = f_scores(ref_int, ref_pitch, *est)

        h = track_hcqt(track, args.cache, secs)
        with torch.no_grad():
            z = encoder.to(_device)(h.to(_device)).float()  # [1, T, D]
            note = head_note(z)[0].sigmoid().cpu().numpy()
            onset = head_onset(z)[0].sigmoid().cpu().numpy()
        est = posteriorgrams_to_notes({"note": note, "onset": onset})
        models["JEPA + heads (ours)"][track["name"]] = f_scores(ref_int, ref_pitch, *est)

    header = f"{'model':<24} {'Fno (main)':>16} {'F':>16} {'Acc':>16}"
    print(header)
    print("-" * len(header))
    for name, per_track in models.items():
        arr = np.array(list(per_track.values()))
        m, s = arr.mean(0), arr.std(0)
        print(
            f"{name:<24} {m[0]:7.3f} ± {s[0]:.3f} {m[1]:7.3f} ± {s[1]:.3f} {m[2]:7.3f} ± {s[2]:.3f}"
        )
    print()
    print("Per-track Fno:")
    for name, per_track in models.items():
        print("  " + name + ": " + " ".join(f"{v[0]:.3f}" for v in per_track.values()))
    print()
    print(f"RESULT eval tracks={len(tracks)} seconds={args.seconds or 'full'}")
    return (models,)


if __name__ == "__main__":
    app.run()
