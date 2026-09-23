# Copyright 2026 AcusPy contributors
#
# Supervised baseline: train the faithful NMP (basic-pitch architecture,
# paper-scale) on the same BabySlakh cache as the JEPA, so both models see
# identical data. This is the "from scratch" reference for the comparison
# against Spotify's pretrained basic-pitch checkpoint.
#
# Run:  python jepa/train_nmp.py [--steps 10 --workers 0]  (smoke test)

import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    import argparse
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    import stable_pretraining as spt
    import lightning as pl
    from lightning.pytorch.loggers import CSVLogger
    from torch import nn

    sys.path.insert(0, str(Path(__file__).parent))
    import babyslakh
    import hcqt
    import nmp_encoder


@app.cell
def _():
    """CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=-1, help="Override epochs for a smoke run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--width-scale", type=int, default=1, help="1 = faithful paper scale")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default="babyslakh_16k")
    parser.add_argument("--cache", default=".cache/babyslakh_hcqt")
    parser.add_argument("--remix-cache", default=None, help="Optional remix CQT cache to extend training data")
    parser.add_argument("--remix-crops", type=int, default=16, help="Crops per remix track per epoch")
    parser.add_argument("--synthetic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--synthetic-samples", type=int, default=512)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="runs")
    args = parser.parse_args([] if mo.running_in_notebook() else None)
    return (args,)


@app.cell
def _(args):
    """Runtime config."""
    spt.set(cache_dir=str(Path(args.output).expanduser().resolve()), requeue_checkpoint=False, verbose="WARNING")
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    return


@app.cell
def _(args):
    """Datasets & loaders (same cache as the JEPA)."""
    from torch.utils.data import DataLoader

    if args.synthetic:
        train_ds = babyslakh.SyntheticClips(args.synthetic_samples, args.frames, args.seed)
        val_ds = babyslakh.SyntheticClips(max(64, args.batch_size), args.frames, args.seed + 1_000_000)
    else:
        babyslakh.ensure_cache(args.data, args.cache, workers=args.workers)
        train_parts = [babyslakh.BabySlakhClips(args.cache, args.frames, "train", 0.2, args.seed)]
        if args.remix_cache:
            remix_ds = babyslakh.BabySlakhClips(
                args.remix_cache, args.frames, "train", 0.0, args.seed, crops_per_track=args.remix_crops
            )
            print(f"remix tracks: {len(remix_ds.names)} x {args.remix_crops} crops")
            train_parts.append(remix_ds)
        train_ds = torch.utils.data.ConcatDataset(train_parts)
        val_ds = babyslakh.BabySlakhClips(args.cache, args.frames, "val", 0.2, args.seed)
    print(f"train clips: {len(train_ds)} | val clips: {len(val_ds)}")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            drop_last=shuffle,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    train_loader, val_loader = make_loader(train_ds, True), make_loader(val_ds, False)
    return (val_loader, train_loader)


@app.cell
def _(args):
    """Model (logits heads) + paper-style class-balanced onset weight."""
    model = nmp_encoder.NMP(width_scale=args.width_scale)
    if args.compile and torch.cuda.is_available():
        model = torch.compile(model, mode="reduce-overhead")
    onset_pos_weight = torch.tensor(19.0)  # 0.95 / 0.05 from the paper
    return (model, onset_pos_weight)


@app.cell
def _(args, model, onset_pos_weight):
    """Lightning module: BCE on contour/note, weighted BCE on onset."""

    def make_targets(batch, t):
        """Expand semitone flags into posteriorgram targets."""
        pitch = batch["pitch"]  # [B, T, 88]
        onset_pitch = batch["onset_pitch"]  # [B, T, 88]
        contour_t = pitch.repeat_interleave(3, dim=-1)[..., :256]  # [B, T, 256]
        note_t = pitch[..., :86]  # 1 bin/semitone
        onset_t = onset_pitch[..., :86]
        return contour_t, note_t, onset_t

    def forward(self, batch, stage):
        hcqt = batch["hcqt"]
        out = self.model(hcqt)
        out = {k: v.transpose(1, 2) for k, v in out.items()}  # [B, T, bins]
        contour_t, note_t, onset_t = make_targets(batch, hcqt.shape[-1])
        loss_contour = nn.functional.binary_cross_entropy_with_logits(
            out["contour"], contour_t
        )
        loss_note = nn.functional.binary_cross_entropy_with_logits(out["note"], note_t)
        loss_onset = nn.functional.binary_cross_entropy_with_logits(
            out["onset"], onset_t, pos_weight=onset_pos_weight.to(out["onset"].device)
        )
        loss = loss_contour + loss_note + loss_onset
        self.log_dict(
            {
                f"{stage}/nmp": loss,
                f"{stage}/contour": loss_contour,
                f"{stage}/note": loss_note,
                f"{stage}/onset": loss_onset,
            },
            on_epoch=True,
        )
        return {"loss": loss}

    module = spt.Module(
        model=model,
        forward=forward,
        hparams=vars(args),
        optim={"optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 1e-4}},
    )
    return (module,)


@app.cell
def _(args, module, train_loader, val_loader):
    """Trainer + manager."""
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1 if args.steps > 0 else args.epochs,
        limit_train_batches=args.steps if args.steps > 0 else 1.0,
        limit_val_batches=4 if args.steps > 0 else 1.0,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[],
        logger=CSVLogger(args.output, name="train_nmp"),
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        seed=args.seed,
    )
    manager()
    return (manager, trainer)


@app.cell
def _(args, manager, module, trainer):
    """Save checkpoint and print the RESULT line."""
    checkpoint = Path(manager._run_dir) / "nmp.pt"
    model_state = {
        key.replace("._orig_mod.", "."): value
        for key, value in module.model.state_dict().items()
    }
    torch.save(
        {
            "model": model_state,
            "config": {"width_scale": args.width_scale, "frames": args.frames},
        },
        checkpoint,
    )
    print(
        f"RESULT train_nmp "
        f"val_contour={float(trainer.callback_metrics['validate/contour']):.4f} "
        f"val_note={float(trainer.callback_metrics['validate/note']):.4f} "
        f"val_onset={float(trainer.callback_metrics['validate/onset']):.4f} "
        f"checkpoint={checkpoint}"
    )
    return (checkpoint,)


if __name__ == "__main__":
    app.run()
