# Choreo2Groove – end‑to‑end pose→drum symbolic generator with CUDA acceleration
"""
Usage (after data prep):
    python choreo2groove.py \ 
        --data_root /path/to/dataset \   # folder with pairs pose.npy & drums.mid
        --epochs 30 --batch_size 16 --seq_len 512 --lr 1e-4

Dataset directory tree (after running scripts/download_aistpp.sh):
    dataset_root/
        sample_000/
            pose.npy          # (T, J, 3) float32, 60 fps
            drums.mid         # symbolic drum track aligned to same T
        sample_001/...
        ...

To build the dataset from raw AIST++ videos:
    1. Run scripts/download_aistpp.sh to grab keypoints & metadata.
    2. Run scripts/extract_pose.py – produces pose.npy in above format.
    3. Run scripts/transcribe_drums.py – uses ADTLib to create drums.mid.
    4. Run scripts/sync_pose_beats.py – trims/quantises both to same tempo.

The core training loop uses PyTorch Lightning for multi‑GPU out‑of‑the‑box.
"""
# pylint: disable=not-callable, no-member, too-many-arguments, line-too-long
import argparse
import math
import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pretty_midi as pm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------
# Constants & token vocabulary
# -------------------------------------------
DRUM_TOKENS = {
    "pad": 0,
    "kick": 1,
    "snare": 2,
    "hihat_closed": 3,
    "hihat_open": 4,
    "tom_low": 5,
    "tom_mid": 6,
    "tom_high": 7,
    "crash": 8,
    "ride": 9,
    # time‑shift tokens (20 ms increments up to 2 s)
}
SHIFT_OFFSET = len(DRUM_TOKENS)
MAX_SHIFT = 100  # 100 * 20 ms = 2 s
for i in range(1, MAX_SHIFT + 1):
    DRUM_TOKENS[f"shift_{i}"] = SHIFT_OFFSET + i
VOCAB_SIZE = len(DRUM_TOKENS)
IDX2TOKEN = {v: k for k, v in DRUM_TOKENS.items()}

# -------------------------------------------
# Utilities
# -------------------------------------------


def midi_to_tokens(mid: pm.PrettyMIDI, time_unit: float = 0.02) -> List[int]:
    """Convert PrettyMIDI drum track to event tokens."""
    events: List[Tuple[float, str]] = []
    for inst in mid.instruments:
        if not inst.is_drum:
            continue
        for note in inst.notes:
            events.append((note.start, _pitch_to_token(note.pitch)))
    events.sort(key=lambda x: x[0])
    tokens = []
    prev_time = 0.0
    for t, tok in events:
        delta = t - prev_time
        n_shift = int(round(delta / time_unit))
        while n_shift > MAX_SHIFT:
            tokens.append(DRUM_TOKENS[f"shift_{MAX_SHIFT}"])
            n_shift -= MAX_SHIFT
        if n_shift > 0:
            tokens.append(DRUM_TOKENS[f"shift_{n_shift}"])
        tokens.append(DRUM_TOKENS[tok])
        prev_time = t
    return tokens


def _pitch_to_token(pitch: int) -> str:
    """Map GM pitch to token name."""
    if pitch in (36, 35):
        return "kick"
    if pitch in (38, 40):
        return "snare"
    if pitch in (42, 44):
        return "hihat_closed"
    if pitch in (46,):
        return "hihat_open"
    if pitch in (45, 41):
        return "tom_low"
    if pitch in (47, 48):
        return "tom_mid"
    if pitch in (50,):
        return "tom_high"
    if pitch in (49, 57):
        return "crash"
    if pitch in (51, 59):
        return "ride"
    return "snare"  # fallback


# -------------------------------------------
# Dataset
# -------------------------------------------
class ChoreoGrooveDataset(Dataset):
    def __init__(self, root: str, seq_len: int = 512):
        self.items = sorted(glob(os.path.join(root, "*", "pose.npy")))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        pose_path = self.items[idx]
        drum_path = pose_path.replace("pose.npy", "drums.mid")
        pose = np.load(pose_path)  # (T, J, 3)
        pose = pose.reshape(pose.shape[0], -1)  # flatten joints
        # compute velocities
        vel = np.diff(pose, axis=0, prepend=pose[:1])
        feats = np.concatenate([pose, vel], axis=-1)
        # normalise
        feats = feats.astype(np.float32)
        feats = (feats - feats.mean()) / (feats.std() + 1e-5)
        # tokenise drums
        midi = pm.PrettyMIDI(drum_path)
        tokens = midi_to_tokens(midi)
        # pad / trim to seq_len
        if len(tokens) < self.seq_len:
            tokens += [DRUM_TOKENS["pad"]] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[: self.seq_len]
        return torch.from_numpy(feats), torch.tensor(tokens, dtype=torch.long)


# -------------------------------------------
# Model
# -------------------------------------------
class PoseEncoder(nn.Module):
    def __init__(self, in_feats: int = 102, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_feats, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, embed_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):  # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # back to (B, T, embed)
        x, _ = self.gru(x)
        x = self.out_proj(x)
        return x.transpose(0, 1)  # (T, B, embed) for Transformer


class DrumDecoder(nn.Module):
    def __init__(self, embed_dim: int = 256, num_layers: int = 4, nhead: int = 8):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.pos_emb = nn.Embedding(1024, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=1024, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, VOCAB_SIZE)

    def forward(self, tgt_tokens, memory):  # tgt: (B, L)
        positions = torch.arange(
            0, tgt_tokens.size(1), device=tgt_tokens.device
        ).unsqueeze(0)
        tgt = self.token_emb(tgt_tokens) + self.pos_emb(positions)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )
        out = self.transformer(tgt, memory.transpose(0, 1), tgt_mask=tgt_mask)
        return self.fc_out(out)


# -------------------------------------------
# Lightning Module
# -------------------------------------------
class Choreo2GrooveModel(pl.LightningModule):
    def __init__(self, in_feats: int, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = PoseEncoder(in_feats)
        self.decoder = DrumDecoder()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=DRUM_TOKENS["pad"])

    def forward(self, poses, drums):
        memory = self.encoder(poses)  # (T, B, E)
        logits = self.decoder(drums[:, :-1], memory)  # predict next token
        return logits

    def training_step(self, batch, batch_idx):
        poses, tokens = batch
        logits = self(poses, tokens)
        loss = self.loss_fn(logits.reshape(-1, VOCAB_SIZE), tokens[:, 1:].reshape(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# -------------------------------------------
# Data utility
# -------------------------------------------


def collate_fn(batch):
    poses, tokens = zip(*batch)
    poses = torch.nn.utils.rnn.pad_sequence(poses, batch_first=True)
    tokens = torch.stack(tokens)
    return poses, tokens


# -------------------------------------------
# CLI
# -------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    ds = ChoreoGrooveDataset(args.data_root, args.seq_len)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    sample_pose, _ = ds[0]
    in_feats = sample_pose.shape[-1]
    model = Choreo2GrooveModel(in_feats, lr=args.lr)

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto")
    trainer.fit(model, dl)


if __name__ == "__main__":
    main()
