# Copyright 2026 AcusPy contributors
#
# LeWM-audio: a temporal-predictive JEPA over NMP features, following the
# LeWM pattern from the stable-pretraining tutorials (utils.py). The encoder
# is the basic-pitch NMP CNN; a light MLP predictor learns to advance the
# per-frame latents in time, and SIGReg keeps the representation from
# collapsing (no EMA teacher needed).
#
# Run interactively with `marimo edit jepa/audio_lewm.py`.

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")

with app.setup:
    import sys
    from pathlib import Path

    import torch
    from torch import nn

    sys.path.insert(0, str(Path(__file__).parent))
    import nmp_encoder
    from stable_pretraining.methods.lejepa import SlicedEppsPulley


@app.class_definition
class AudioLeWM(nn.Module):
    """Temporal-predictive JEPA on NMP features.

    Input:  hcqt [B, 8, 256, T]  ->  z [B, T, D]  per-frame latents
    Loss:   one-step prediction + weighted open-loop rollout + SIGReg.
    """

    def __init__(
        self,
        embed_dim=64,
        hidden_dim=256,
        width_scale=8,
        lamb=0.1,
        slices=1024,
        rollout_weight=1.0,
        sigreg_mode="pooled",
    ):
        super().__init__()
        self.encoder = nmp_encoder.NMPTrunk(width_scale, embed_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.sigreg = SlicedEppsPulley(num_slices=slices)
        self.lamb = lamb
        self.rollout_weight = rollout_weight
        self.sigreg_mode = sigreg_mode

    def step(self, latent):
        return self.predictor(latent)

    def forward(self, hcqt):
        """Teacher-forced training loss for [B, 8, 256, T] clips.

        Returns (loss, prediction, sigreg, z).
        """
        z = self.encoder(hcqt).clone()  # [B, T, D]
        one_step = self.step(z[:, :-1]).clone()
        state, rollout = z[:, 0], []
        for t in range(z.shape[1] - 1):
            state = self.step(state).clone()
            rollout.append(state)
        rollout = torch.stack(rollout, dim=1)
        pred_loss = (one_step - z[:, 1:]).square().mean()
        pred_loss = pred_loss + self.rollout_weight * (rollout - z[:, 1:]).square().mean()
        pooled = self.sigreg(z.flatten(0, 1))
        if self.sigreg_mode == "pooled":
            sigreg = pooled
        elif self.sigreg_mode == "per_time":
            sigreg = torch.stack([self.sigreg(z[:, t]) for t in range(z.shape[1])]).mean()
        elif self.sigreg_mode == "both":
            per_time = torch.stack([self.sigreg(z[:, t]) for t in range(z.shape[1])]).mean()
            sigreg = 0.5 * (pooled + per_time)
        elif self.sigreg_mode == "pooled_pred":
            predicted = self.sigreg(rollout.flatten(0, 1))
            sigreg = 0.5 * (pooled + predicted)
        else:
            raise ValueError(f"unknown SIGReg mode: {self.sigreg_mode}")
        return pred_loss + self.lamb * sigreg, pred_loss, sigreg, z

    @torch.no_grad()
    def rollout(self, hcqt):
        """Open-loop latent rollout from the first frame: [B, T, D]."""
        z = self.encoder(hcqt)
        out = [z[:, 0]]
        for _ in range(z.shape[1] - 1):
            out.append(self.step(out[-1]))
        return torch.stack(out, dim=1)


@app.cell
def _():
    """Doc cell."""
    import marimo as mo

    mo.md(
        """
        # AudioLeWM

        - **Encoder**: trunk de la CNN NMP (basic-pitch), latents por frame.
        - **Predictor**: MLP z_t -> z_{t+1}; pérdida one-step + rollout open-loop.
        - **SIGReg** evita el colapso (misma família que LeJEPA/LeWM).
        """
    )
    return


@app.cell
def _():
    """Smoke test: forward/backward on GPU or CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioLeWM(width_scale=4, slices=64).to(device)
    hcqt = torch.rand(2, 8, 256, 48, device=device)
    loss, pred, sig, z = model(hcqt)
    assert z.shape == (2, 48, 64), z.shape
    loss.backward()
    loss, pred, sig = loss.detach(), pred.detach(), sig.detach()
    print(f"device={device} loss={float(loss):.4f} pred={float(pred):.4f} sigreg={float(sig):.4f}")
    return


if __name__ == "__main__":
    app.run()
