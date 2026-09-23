# Copyright 2026 AcusPy contributors
#
# Train a temporal-predictive JEPA (LeWM) on NMP features over BabySlakh.
#
# Run:
#   python jepa/train_audio.py --synthetic --steps 10 --workers 0   # smoke test
#   python jepa/train_audio.py                                      # full run
# or edit interactively:
#   marimo edit jepa/train_audio.py

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
    import torchmetrics
    import stable_pretraining as spt
    import lightning as pl
    from lightning.pytorch.loggers import CSVLogger
    from torch import nn

    sys.path.insert(0, str(Path(__file__).parent))
    import babyslakh
    import audio_lewm


@app.cell
def _():
    """CLI arguments (defaults when running inside marimo)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=-1, help="Override epochs for a smoke run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--frames", type=int, default=96, help="HCQT frames per clip (~2.2s)")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--width-scale", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe-lr", type=float, default=3e-3)
    parser.add_argument("--lamb", type=float, default=0.1)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument(
        "--sigreg-mode",
        choices=("pooled", "per_time", "both", "pooled_pred"),
        default="pooled",
    )
    parser.add_argument("--slices", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default="babyslakh_16k", help="BabySlakh/Slakh root")
    parser.add_argument("--cache", default=".cache/babyslakh_hcqt", help="CQT cache dir")
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
    """Reproducibility & runtime config."""
    spt.set(cache_dir=str(Path(args.output).expanduser().resolve()), requeue_checkpoint=False, verbose="WARNING")
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    return


@app.cell
def _(args):
    """Datasets & loaders: synthetic for smoke tests, BabySlakh for real runs."""
    from torch.utils.data import DataLoader

    if args.synthetic:
        train_ds = babyslakh.SyntheticClips(args.synthetic_samples, args.frames, args.seed)
        val_ds = babyslakh.SyntheticClips(max(64, args.batch_size), args.frames, args.seed + 1_000_000)
    else:
        names = babyslakh.ensure_cache(args.data, args.cache, workers=args.workers)
        print(f"cached tracks: {len(names)}")
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
    """Model (compile on CUDA for speed)."""
    model = audio_lewm.AudioLeWM(
        embed_dim=args.latent_dim,
        width_scale=args.width_scale,
        lamb=args.lamb,
        slices=args.slices,
        rollout_weight=args.rollout_weight,
        sigreg_mode=args.sigreg_mode,
    )
    if args.compile and torch.cuda.is_available():
        model.encoder = torch.compile(model.encoder, mode="reduce-overhead")
        model.predictor = torch.compile(model.predictor, mode="reduce-overhead")
    return (model,)


@app.cell
def _(args, model):
    """Lightning module wiring (mirrors mmnist.py)."""

    def forward(self, batch, stage):
        loss, pred, sig, z = self.model(batch["hcqt"])
        self.log_dict(
            {f"{stage}/lewm": loss, f"{stage}/prediction": pred, f"{stage}/sigreg": sig},
            on_epoch=True,
        )
        return {
            "loss": loss,
            "embedding": z.flatten(0, 1),
            "pitch_target": batch["pitch"].argmax(-1).flatten().long(),
            "onset_target": batch["onset"].flatten(0, 1)[:, None].float(),
        }

    module = spt.Module(
        model=model,
        forward=forward,
        hparams=vars(args),
        optim={
            "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 1e-4},
            "scheduler": {"type": "CosineAnnealingLR"},
            "interval": "step",
        },
    )
    return (module,)


@app.cell
def _(args, module):
    """Online probes: pitch (88 classes) and onset, linear + MLP."""
    pitch_probe = spt.callbacks.OnlineProbe(
        module,
        "pitch_probe",
        "embedding",
        "pitch_target",
        nn.Linear(args.latent_dim, 88),
        loss=nn.CrossEntropyLoss(),
        optimizer={"type": "AdamW", "lr": args.probe_lr, "weight_decay": 1e-7},
        metrics={"acc": torchmetrics.classification.MulticlassAccuracy(88)},
    )
    pitch_mlp_probe = spt.callbacks.OnlineProbe(
        module,
        "pitch_mlp_probe",
        "embedding",
        "pitch_target",
        nn.Sequential(
            nn.Linear(args.latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 88),
        ),
        loss=nn.CrossEntropyLoss(),
        optimizer={"type": "AdamW", "lr": args.probe_lr, "weight_decay": 1e-7},
        metrics={"acc": torchmetrics.classification.MulticlassAccuracy(88)},
    )
    onset_probe = spt.callbacks.OnlineProbe(
        module,
        "onset_probe",
        "embedding",
        "onset_target",
        nn.Linear(args.latent_dim, 1),
        loss=nn.BCEWithLogitsLoss(),
        optimizer={"type": "AdamW", "lr": args.probe_lr, "weight_decay": 1e-7},
        metrics={"acc": torchmetrics.classification.BinaryAccuracy()},
    )
    return (onset_probe, pitch_mlp_probe, pitch_probe)


@app.cell
def _(args, module, onset_probe, pitch_mlp_probe, pitch_probe, train_loader, val_loader):
    """Trainer + stable-pretraining manager."""
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1 if args.steps > 0 else args.epochs,
        limit_train_batches=args.steps if args.steps > 0 else 1.0,
        limit_val_batches=4 if args.steps > 0 else 1.0,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[pitch_probe, pitch_mlp_probe, onset_probe],
        logger=CSVLogger(args.output, name="train_audio"),
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
def _(args, manager, trainer, module):
    """Save a flat checkpoint and print the RESULT line."""
    checkpoint = Path(manager._run_dir) / "lewm_audio.pt"
    model_state = {
        key.replace("._orig_mod.", "."): value
        for key, value in module.model.state_dict().items()
    }
    torch.save(
        {
            "model": model_state,
            "config": {
                "latent_dim": args.latent_dim,
                "width_scale": args.width_scale,
                "lamb": args.lamb,
                "slices": args.slices,
                "rollout_weight": args.rollout_weight,
                "sigreg_mode": args.sigreg_mode,
                "frames": args.frames,
            },
        },
        checkpoint,
    )
    pitch_acc = float(trainer.callback_metrics["eval/pitch_probe_acc"])
    pitch_mlp_acc = float(trainer.callback_metrics["eval/pitch_mlp_probe_acc"])
    onset_acc = float(trainer.callback_metrics["eval/onset_probe_acc"])
    pred_mse = float(trainer.callback_metrics["validate/prediction"])
    print(
        f"RESULT train_audio pitch_linear_acc={pitch_acc:.4f} "
        f"pitch_mlp_acc={pitch_mlp_acc:.4f} onset_acc={onset_acc:.4f} "
        f"rollout_prediction_mse={pred_mse:.6f} checkpoint={checkpoint}"
    )
    return (checkpoint,)


if __name__ == "__main__":
    app.run()
