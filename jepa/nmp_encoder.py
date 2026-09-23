# Copyright 2026 AcusPy contributors
#
# PyTorch port of the NMP encoder from Bittner et al. (2022), "A Lightweight
# Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch
# Estimation" (basic-pitch). Kernels, strides and stage order are faithful;
# channel widths are scaled by `width_scale` for self-supervised training.
#
# Run interactively with `marimo edit jepa/nmp_encoder.py` or
# `import nmp_encoder` from other notebooks.

import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    import torch
    from torch import nn


@app.function
class NMPTrunk(nn.Module):
    """Feature trunk of the NMP CNN, faithful to basic-pitch's model.

    Input:  HCQT  [B, 8, 256, T]  (harmonic-stacked log CQT)
    Output: z     [B, T', D]      per-frame latents (T' ≈ T/3)

    Stage order mirrors basic-pitch (TF layout (time, freq) -> PyTorch (freq, time)):
      conv (39,3) -> BN -> ReLU                 (contour features; 39 = octave+semitone)
      conv (5,5)  -> BN -> ReLU                 (contour-resolution features)
      conv (7,7)  -> BN -> ReLU, stride (3,1)   (musical quantization features)
      conv (3,7)  -> BN -> ReLU                 (note-head features)
      frequency pooling + linear projection
    """

    def __init__(self, width_scale=8, embed_dim=64):
        super().__init__()
        c1 = c2 = 8 * width_scale
        c3 = 32 * width_scale
        c4 = 16 * width_scale
        self.conv_contour = nn.Sequential(
            nn.Conv2d(8, c1, (39, 3), padding=(19, 1)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, (5, 5), padding=2),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.conv_notes = nn.Sequential(
            nn.Conv2d(c2, c3, (7, 7), padding=(3, 3), stride=(3, 1)),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, (3, 7), padding=(1, 3)),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Linear(c4, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, hcqt):
        """hcqt: [B, 8, 256, T] -> z: [B, T, embed_dim]."""
        x = self.conv_contour(hcqt)
        x = self.conv_notes(x)  # [B, c4, F', T], F' = 86
        x = x.mean(dim=2)  # pool frequency -> [B, c4, T]
        z = self.project(x.transpose(1, 2))  # [B, T, D]
        return z


@app.function
class NMP(nn.Module):
    """Full NMP with the paper's three posteriorgram heads.

    contour: multipitch, 3 bins/semitone at 256-bin resolution
    note:    note activity, 1 bin/semitone (86 bins)
    onset:   onsets, 1 bin/semitone, from note features + audio features
    """

    def __init__(self, width_scale=1):
        super().__init__()
        c1 = c2 = 8 * width_scale
        c3 = 32 * width_scale
        c5 = 32 * width_scale
        self.conv_contour = nn.Sequential(
            nn.Conv2d(8, c1, (39, 3), padding=(19, 1)),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, (5, 5), padding=2),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.contour_head = nn.Conv2d(c2, 1, (5, 5), padding=2)
        self.conv_notes = nn.Sequential(
            nn.Conv2d(1, c3, (7, 7), padding=(3, 3), stride=(3, 1)),
            nn.ReLU(inplace=True),
        )
        self.note_head = nn.Conv2d(c3, 1, (3, 7), padding=(1, 3))
        self.conv_onset = nn.Sequential(
            nn.Conv2d(8, c5, (5, 5), padding=(2, 2), stride=(3, 1)),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
        )
        self.onset_head = nn.Conv2d(c5 + 1, 1, (3, 3), padding=1)

    def forward(self, hcqt):
        """hcqt: [B, 8, 256, T] -> dict of posteriorgram LOGITS.

        Apply .sigmoid() for posteriorgrams at inference time.
        """
        x = self.conv_contour(hcqt)
        contour = self.contour_head(x)  # [B, 1, 256, T]
        x_notes = self.conv_notes(contour.sigmoid())  # [B, c3, 86, T]
        note = self.note_head(x_notes)  # [B, 1, 86, T]
        x_onset = self.conv_onset(hcqt)  # [B, c5, 86, T]
        onset = self.onset_head(
            torch.cat((note.sigmoid(), x_onset), dim=1)
        )  # [B, 1, 86, T]
        return {
            "contour": contour.squeeze(1),
            "note": note.squeeze(1),
            "onset": onset.squeeze(1),
        }


@app.cell
def _():
    """Param counts per width scale (faithful NMP = 16,782 at scale 1)."""
    def count(module):
        return sum(p.numel() for p in module.parameters())

    nmp = NMP(width_scale=1)
    print("NMP (paper-scale):", count(nmp), "params")
    for scale in (1, 4, 8, 16):
        print(f"NMPTrunk(width_scale={scale}):", count(NMPTrunk(scale)), "params")
    return


@app.cell
def _():
    """Shape smoke test."""
    hcqt = torch.randn(2, 8, 256, 96)
    z = NMPTrunk(width_scale=8, embed_dim=64)(hcqt)
    assert z.shape == (2, 96, 64), z.shape
    out = NMP(width_scale=1)(hcqt)
    assert out["contour"].shape == (2, 256, 96), out["contour"].shape
    assert out["note"].shape == (2, 86, 96), out["note"].shape
    assert out["onset"].shape == (2, 86, 96), out["onset"].shape
    print("shapes OK: z", tuple(z.shape), "| heads:", {k: tuple(v.shape) for k, v in out.items()})
    return


if __name__ == "__main__":
    app.run()
