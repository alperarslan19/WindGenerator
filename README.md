# WindGenerator

A deep learning system that generates realistic wind audio from scratch using a diffusion model and a custom neural vocoder trained with GAN fine-tuning.

**Pipeline:** Gaussian noise → DDPM diffusion model → mel spectrogram → GAN vocoder → wind audio

---

## Demo

> *Generated wind audio samples will be added here after final model training.*

---

## Overview

WindGenerator is an end-to-end generative audio pipeline built entirely from scratch. The system learns the statistical structure of wind sound from a dataset of real recordings, and can generate novel wind audio clips that were never in the training data.

The project was motivated by a simple observation: wind is one of the most common ambient sounds in games, film, and interactive media, yet high-quality procedural wind generation remains a hard problem. Rather than using hand-crafted signal processing, this project trains neural networks to learn what wind sounds like directly from data.

---

## Architecture

The system has two main components trained independently.

### 1. Diffusion Model (DDPM)

A UNet-based denoising diffusion probabilistic model trained on log-mel spectrograms. Given a random Gaussian noise tensor, the model iteratively denoises it over 1000 timesteps to produce a clean mel spectrogram representing the spectral structure of wind.

- **Input:** Gaussian noise `(1, 128, 440)`
- **Output:** Normalized log-mel spectrogram `(1, 128, 440)`
- **Architecture:** UNet with 3 resolution levels, channels `(64, 128, 256)`, `~10M` parameters
- **Training:** 100,000 steps on 1,966 wind audio clips

The mel spectrogram format encodes 5.12 seconds of audio at 22,050 Hz using 128 mel frequency bins and a hop length of 256 samples.

### 2. Neural Vocoder (TinyVocoder + GAN)

A custom lightweight neural vocoder that converts mel spectrograms to raw audio waveforms. Training proceeded in two phases:

**Phase 1 — STFT Loss Only (75,000 steps)**

The vocoder learns the mapping from mel to waveform using multi-resolution STFT loss computed at three scales `(512, 1024, 2048)`. This phase establishes correct spectral structure but produces waveforms with phase artifacts — a characteristic "buzz" caused by phase averaging.

**Phase 2 — GAN Fine-tuning (50,000 steps)**

A Combined Discriminator (MPD + MSD) is introduced to push the generator toward perceptually natural outputs.

- **Multi-Period Discriminator (MPD):** 5 sub-discriminators with periods `[2, 3, 5, 7, 11]` operating on 2D reshapes of the waveform. Detects periodic artifacts at multiple timescales.
- **Multi-Scale Discriminator (MSD):** 3 sub-discriminators operating on the raw waveform at `1×`, `2×`, and `4×` average-pooled resolutions. Captures amplitude envelope and spectral shape at different temporal resolutions.

Generator loss during GAN training:

```
g_loss = 45.0 × stft_loss + 1.0 × adv_loss + 2.0 × fm_loss
```

The 45× STFT weight is calibrated from the HiFi-GAN paper to ensure spectral quality gradients always dominate over adversarial gradients — preventing the generator from trading spectral accuracy for adversarial performance.

**Vocoder architecture:**

```
Input: (B, 128, 440) normalized mel
  └── Conv1d(128 → 256, k=7)
  └── UpBlock ×8   [interpolate + Conv1d, MRFBlock]
  └── UpBlock ×8   [interpolate + Conv1d, MRFBlock]
  └── UpBlock ×4   [interpolate + Conv1d, MRFBlock]
  └── Conv1d → tanh
Output: (B, 112640) waveform  [= 5.12s at 22050 Hz]
```

Each MRFBlock contains 3 parallel residual branches with kernels `[3, 7, 11]` and dilations `[1, 3, 5]`, capturing multi-scale temporal structure. Total upsample factor: `8 × 8 × 4 = 256 = hop_length`. ✓

---

## Training Challenges and Solutions

This project involved significant debugging of training dynamics. The key challenges encountered and solved:

### MPS/GPU Compatibility
Apple Silicon MPS backend caused crashes with `torch.stft`, `F.interpolate`, and `ConvTranspose1d`. Solution: replaced `ConvTranspose1d` with `F.interpolate(mode='nearest') + Conv1d` for artifact-free upsampling, and migrated training to CUDA on Kaggle/Colab.

### STFT Loss NaN Gradients
`torch.abs()` on complex STFT output produces undefined gradients at zero magnitude. Solution: replaced with `sqrt(Re² + Im² + 1e-9)` for numerically stable magnitude computation.

### GAN Training Instability (Asymmetric Warmup)
Initial GAN training used a 1000-step warmup where the discriminator trained uncontested while the generator received no adversarial gradient. The discriminator became overconfident before adversarial training began, causing the generator to sacrifice spectral quality when GAN activated. 

Solution (derived from HiFi-GAN analysis):
- Eliminated warmup entirely — generator already pretrained from Phase 1
- Changed LR scheduler from per-step to per-epoch decay (`gamma=0.999`)
- Increased STFT loss weight to `45.0` matching HiFi-GAN's calibrated value
- Fixed discriminator weight normalization bug (missing `weight_norm` on Conv2d layers)

### Checkpoint Persistence
Both Kaggle and Google Colab wipe working directories on session end. Solution: implemented automatic Drive backup after every checkpoint save, with restore-on-startup logic allowing seamless session resumption.

---

## Repository Structure

```
WindGenerator/
├── src/windgen/
│   ├── mels.py              # LogMelExtractor with validation assertions
│   ├── dataset.py           # WindMelDataset
│   └── vocoder_tiny.py      # TinyVocoder, LiteMPD, LiteMSD, CombinedDisc, STFT loss
├── scripts/
│   ├── prepare_dataset.py   # Raw audio → fixed-length clips
│   ├── compute_mel_stats.py # Global mel normalization statistics
│   ├── train_diffusion.py   # DDPM training
│   ├── train_vocoder_stft.py # Phase 1 vocoder training
│   ├── train_vocoder_gan.py  # Phase 2 GAN fine-tuning
│   └── generate_audio.py    # End-to-end generation
└── outputs/
    ├── mel_stats.json        # Global normalization stats
    ├── train_ddpm/           # Diffusion checkpoints
    └── train_vocoder/        # Vocoder checkpoints
```

---

## Dataset

1,966 wind audio clips, each 5.12 seconds, 22,050 Hz mono. Clips were segmented from longer recordings and filtered to remove silence and non-wind content.

**Mel spectrogram configuration:**

| Parameter | Value |
|---|---|
| Sample rate | 22,050 Hz |
| FFT size | 1,024 |
| Hop length | 256 |
| Window length | 1,024 |
| Mel bins | 128 |
| Frequency range | 20 Hz – 11,025 Hz |

Normalization: log mel → global z-score → clamp ±4σ → scale to `[-1, 1]`

---

## Installation

```bash
git clone https://github.com/alpercagan/WindGenerator.git
cd WindGenerator
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.0+, torchaudio, diffusers, soundfile

---

## Usage

### Generate Wind Audio

```bash
python scripts/generate_audio.py \
    --diffusion_ckpt outputs/train_ddpm/final_model.pt \
    --vocoder_ckpt outputs/train_vocoder/latest_checkpoint.pt \
    --mel_stats outputs/mel_stats.json \
    --num_clips 3 \
    --ddpm_steps 50
```

### Train From Scratch

```bash
# 1. Prepare dataset
python scripts/prepare_dataset.py --input_dir /path/to/raw_audio

# 2. Compute mel statistics
python scripts/compute_mel_stats.py

# 3. Train diffusion model
python scripts/train_diffusion.py --max_steps 100000

# 4. Train vocoder — Phase 1 (STFT)
python scripts/train_vocoder_stft.py --max_steps 100000

# 5. Train vocoder — Phase 2 (GAN)
python scripts/train_vocoder_gan.py \
    --stft_ckpt outputs/train_vocoder_stft/final_model.pt \
    --max_steps 50000
```

---

## Results

| Phase | Steps | STFT Loss | Audio Quality |
|---|---|---|---|
| Vocoder Phase 1 (STFT only) | 75,000 | 0.871 | Wind texture correct, phase buzz present |
| Vocoder Phase 2 (GAN) | 15,000+ | 0.919 | Reduced buzz, more natural texture |
| Diffusion model | 100,000* | — | Generates varied wind spectrograms |

*Diffusion model retraining in progress with upgraded architecture.

---

## References

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (NeurIPS 2020)
- Kong et al., [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) (NeurIPS 2020)
- Kumar et al., [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) (NeurIPS 2019)

---

## Author

Alper Cagan — built as a portfolio project demonstrating generative audio, diffusion models, and adversarial training from scratch.
