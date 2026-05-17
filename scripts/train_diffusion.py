#!/usr/bin/env python3
"""
train_diffusion.py

MPS-friendly DDPM training on log-mel "images".

Key features:
- Fixed mel shape: (1, 128, 440)
- Attention-free UNet mid-block (no attention)
- ~11M parameter UNet + gradient checkpointing
- Batch size = 1 with gradient accumulation (stable on Apple MPS)
- Periodic sampling (saves mel grids as PNG)
- Periodic checkpoint saving

Run:
  python scripts/train_diffusion.py

Outputs:
  outputs/train_ddpm/
    samples_step_000100.png ...
    ckpt_step_001000.pt ...
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMScheduler

from windgen.dataset import WindMelDataset, DatasetConfig
from windgen.mels import MelSpecConfig
from windgen.viz import save_mel_grid


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train DDPM on wind log-mel spectrograms")
    ap.add_argument("--data_dir", type=str, default="/root/Datasets/wind_clean",
                    help="Path to wind_clean dataset directory")
    ap.add_argument("--mel_stats", type=str, default="outputs/mel_stats.json",
                    help="Path to mel_stats.json (absolute or relative to repo root)")
    ap.add_argument("--output_dir", type=str, default="outputs/train_ddpm",
                    help="Directory to write checkpoints and samples")
    ap.add_argument("--max_steps", type=int, default=100_000,
                    help="Total training steps")
    ap.add_argument("--drive_dir", type=str, default=None,
                    help="If provided, copy each checkpoint here after saving")
    return ap.parse_args()


def main():
    args = parse_args()

    # -------------------------
    # Paths / device
    # -------------------------
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_dir = Path(args.drive_dir).resolve() if args.drive_dir else None
    if drive_dir:
        drive_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    # Optional stability tweak (safe)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # -------------------------
    # Mel + dataset config
    # -------------------------
    mel_cfg = MelSpecConfig(
        sr=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=128,
    )
    target_frames = 440  # divisible by 8 for 3 downsamples

    ds = WindMelDataset(
        DatasetConfig(data_dir=data_dir, mel_stats_relpath=args.mel_stats),
        mel_cfg,
        target_frames=target_frames,
    )

    # -------------------------
    # Loader (MPS-friendly)
    # -------------------------
    batch_size = 1
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # -------------------------
    # Model (small + no attention)
    # -------------------------
    H, W = 128, target_frames

    # ~11M parameter UNet — 3 levels to keep 440-wide input divisible by 2³=8.
    # block_out_channels=(64,128,256) with layers_per_block=2 is the minimum
    # config that comfortably exceeds 10M params while staying attention-free.
    model = UNet2DModel(
        sample_size=(H, W),
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        mid_block_type="UNetMidBlock2D",  # no attention
        norm_num_groups=8,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"UNet parameters: {total_params:,}")

    # Big memory saver on MPS (compute-heavy but worth it)
    model.enable_gradient_checkpointing()

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # -------------------------
    # Optim / training config
    # -------------------------
    lr = 2e-4
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    steps = args.max_steps
    grad_accum_steps = 8       # effective batch ~ 8
    log_every = 50
    sample_every = 500
    save_every = 1000

    # Sampling config (keep small for memory)
    sample_batch = 2

    model.train()
    global_step = 0
    micro_step = 0  # counts accumulation steps

    optim.zero_grad(set_to_none=True)

    pbar = tqdm(total=steps, desc="Training", unit="step")
    dl_iter = iter(dl)

    while global_step < steps:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        x0 = batch["mel"].to(device)  # (1,1,128,440)
        bsz = x0.shape[0]

        # Random timesteps per sample
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()

        # Add noise
        noise = torch.randn_like(x0)
        xt = noise_scheduler.add_noise(x0, noise, timesteps)

        # Predict noise
        pred = model(xt, timesteps).sample

        # MSE loss
        loss = torch.mean((pred - noise) ** 2)

        # Keep unscaled loss for logging
        loss_item = float(loss.detach().cpu())

        # Gradient accumulation
        (loss / grad_accum_steps).backward()
        micro_step += 1

        # Only update weights every grad_accum_steps
        if micro_step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1
            pbar.update(1)

            if global_step % log_every == 0:
                pbar.set_postfix(loss=loss_item)

            # Save sample images
            if global_step % sample_every == 0:
                model.eval()
                with torch.no_grad():
                    x = torch.randn((sample_batch, 1, H, W), device=device)
                    for t in noise_scheduler.timesteps:
                        out = model(x, t).sample
                        x = noise_scheduler.step(out, t, x).prev_sample
                    save_mel_grid(
                        x,
                        out_dir / f"samples_step_{global_step:06d}.png",
                        title=f"DDPM samples @ step {global_step}",
                    )
                model.train()

            # Save checkpoints
            if global_step % save_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "step": global_step,
                    "mel_cfg": mel_cfg.__dict__,
                    "target_frames": target_frames,
                }
                ckpt_path = out_dir / f"ckpt_step_{global_step:06d}.pt"
                torch.save(ckpt, ckpt_path)
                if drive_dir:
                    shutil.copy2(ckpt_path, drive_dir / ckpt_path.name)

    pbar.close()
    torch.save(
        {
            "model": model.state_dict(),
            "step": global_step,
            "mel_cfg": mel_cfg.__dict__,
            "target_frames": target_frames,
        },
        out_dir / "final_model.pt",
    )
    print("Done. Outputs in:", out_dir)


if __name__ == "__main__":
    main()