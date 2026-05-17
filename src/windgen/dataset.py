from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from windgen.mels import MelSpecConfig, LogMelExtractor, load_wav_mono_resample


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: Path  # directory containing .wav clips (and optionally metadata.csv)
    # Path A: global stats file used for normalization (absolute or relative to repo root)
    mel_stats_relpath: str = "outputs/mel_stats.json"


class WindMelDataset(Dataset):
    """
    Dataset that returns globally-normalized log-mel tensors in ~[-1, 1].

    Path A contract:
    - logmel = log(mel + eps)
    - z = (logmel - global_mean) / global_std
    - x = clamp(z, -4, 4) / 4
    """

    def __init__(self, cfg: DatasetConfig, mel_cfg: MelSpecConfig, target_frames: int = 440):
        self.cfg = cfg
        self.data_dir = Path(os.path.expanduser(str(cfg.data_dir))).resolve()

        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

        meta_path = self.data_dir / "metadata.csv"
        if meta_path.exists():
            self.df = pd.read_csv(meta_path)
            if "clip_filename" not in self.df.columns:
                raise ValueError("metadata.csv must contain clip_filename column")
        else:
            wav_files = sorted(self.data_dir.glob("*.wav"))
            if not wav_files:
                raise FileNotFoundError(
                    f"No metadata.csv and no *.wav files found in: {self.data_dir}"
                )
            self.df = pd.DataFrame({"clip_filename": [f.name for f in wav_files]})
            print(f"  No metadata.csv; discovered {len(wav_files)} wav files in {self.data_dir}")

        # Resolve repo root: .../WindGenerator/src/windgen/dataset.py -> parents[2] = repo root
        repo_root = Path(__file__).resolve().parents[2]
        stats_path = (repo_root / cfg.mel_stats_relpath).resolve()
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Missing mel stats json: {stats_path}\n"
                f"Expected at repo_root/{cfg.mel_stats_relpath}. "
                f"Create it (or copy it) before training."
            )

        # Path A: global normalization using mel_stats.json
        self.extractor = LogMelExtractor(
            mel_cfg,
            device="cpu",  # dataset runs on CPU; training can move tensors to MPS/CUDA later
            normalization="global",
            stats_path=str(stats_path),
            clamp=4.0,
        )

        self.target_frames = int(target_frames)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        clip_name = str(row["clip_filename"])
        clip_path = self.data_dir / clip_name

        wav = load_wav_mono_resample(clip_path, target_sr=self.extractor.config.sr)

        # mel is normalized log-mel in ~[-1, 1], shape (1, n_mels, T)
        mel = self.extractor(wav)

        # Force fixed time frames (crop or pad) to avoid UNet size mismatches
        T = mel.shape[-1]
        if T > self.target_frames:
            mel = mel[..., : self.target_frames]
        elif T < self.target_frames:
            mel = F.pad(mel, (0, self.target_frames - T), mode="constant", value=0.0)

        return {
            "mel": mel,
            "clip_filename": clip_name,
            "source_path": str(row.get("source_path", "")),
        }