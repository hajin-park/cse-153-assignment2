# Choreo2Groove – end‑to‑end pose→drum symbolic generator
# (patched 2025‑06‑03)

import argparse, math, os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np, pretty_midi as pm, torch
import torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

SHIFT_SEC = 0.02
DRUM_TOKENS: Dict[str, int] = {
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
}
# time‑shift tokens (20 ms each, up to 2 s)
SHIFT_OFFSET = len(DRUM_TOKENS)
MAX_SHIFT = 100  # 100 × 20 ms  = 2 s
for i in range(1, MAX_SHIFT + 1):
    DRUM_TOKENS[f"shift_{i}"] = SHIFT_OFFSET + i

# sequence control tokens – **added**
DRUM_TOKENS["bos"] = len(DRUM_TOKENS)  # begin‑of‑sequence
DRUM_TOKENS["eos"] = len(DRUM_TOKENS)  # end‑of‑sequence

VOCAB_SIZE = len(DRUM_TOKENS)
IDX2TOKEN = {v: k for k, v in DRUM_TOKENS.items()}
PAD_IDX = DRUM_TOKENS["pad"]
BOS_IDX = DRUM_TOKENS["bos"]
EOS_IDX = DRUM_TOKENS["eos"]


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def midi_to_tokens(mid: pm.PrettyMIDI, time_unit: float = SHIFT_SEC) -> List[int]:
    """Drum MIDI → event tokens (no BOS/EOS)."""
    events: List[Tuple[float, str]] = []
    for inst in mid.instruments:
        if not inst.is_drum:
            continue
        for note in inst.notes:
            events.append((note.start, _pitch_to_token(note.pitch)))
    events.sort(key=lambda x: x[0])

    tokens, prev_time = [], 0.0
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


def _pitch_to_token(p: int) -> str:
    # General MIDI → symbolic token
    return (
        "kick"
        if p in (35, 36)
        else (
            "snare"
            if p in (38, 40)
            else (
                "hihat_closed"
                if p in (42, 44)
                else (
                    "hihat_open"
                    if p == 46
                    else (
                        "tom_low"
                        if p in (41, 45)
                        else (
                            "tom_mid"
                            if p in (47, 48)
                            else (
                                "tom_high"
                                if p == 50
                                else (
                                    "crash"
                                    if p in (49, 57)
                                    else "ride" if p in (51, 59) else "snare"
                                )
                            )
                        )
                    )
                )
            )
        )
    )


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class ChoreoGrooveDataset(Dataset):
    def __init__(self, root: str, seq_len: int = 512):
        self.items = sorted(glob(os.path.join(root, "*", "pose.npy")))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        pose_path = self.items[idx]
        drum_path = pose_path.replace("pose.npy", "drums.mid")

        # pose → features
        pose = np.load(pose_path).reshape(-1, 51)  # (T, 17×3)
        vel = np.diff(pose, axis=0, prepend=pose[:1])
        feats = np.concatenate([pose, vel], axis=-1)  # (T, 102)
        feats = (feats - feats.mean()) / (feats.std() + 1e-5)
        feats = feats.astype(np.float32)

        # drums → tokens  [+ BOS/EOS, pad/trim]
        tokens = [BOS_IDX] + midi_to_tokens(pm.PrettyMIDI(drum_path)) + [EOS_IDX]
        if len(tokens) < self.seq_len:
            tokens += [PAD_IDX] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[: self.seq_len]

        return torch.from_numpy(feats), torch.tensor(tokens, dtype=torch.long)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class PoseEncoder(nn.Module):
    def __init__(self, in_feats=102, embed=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_feats, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, embed, 3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(embed, embed, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(embed * 2, embed)

    def forward(self, x):  # x (B,T,F)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B,T,E)
        x, _ = self.gru(x)
        return self.proj(x).transpose(0, 1)  # (T,B,E)


class DrumDecoder(nn.Module):
    def __init__(self, embed=256, layers=4, nhead=8, vocab=VOCAB_SIZE):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, embed)
        self.pos_emb = nn.Embedding(1024, embed)
        dec_layer = nn.TransformerDecoderLayer(embed, nhead, 1024, batch_first=True)
        self.transformer = nn.TransformerDecoder(dec_layer, layers)
        self.fc_out = nn.Linear(embed, vocab)

    def forward(self, tgt, memory):  # tgt (B,L), memory (T,B,E)
        pos = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        tgt = self.tok_emb(tgt) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )
        out = self.transformer(tgt, memory.transpose(0, 1), tgt_mask=mask)
        return self.fc_out(out)  # (B,L,V)


class Choreo2GrooveModel(pl.LightningModule):
    def __init__(self, in_feats: int, lr=1e-4):
        super().__init__()
        self.encoder = PoseEncoder(in_feats)
        self.decoder = DrumDecoder()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.save_hyperparameters()

    # ---------- forward ----------
    def forward(self, poses, tokens):
        memory = self.encoder(poses)  # (T,B,E)
        if self.training:
            tgt_in = tokens[:, :-1]  # strip last (EOS / PAD)
            return self.decoder(tgt_in, memory)  # (B,L‑1,V)
        else:
            return self.decoder(tokens, memory)

    # ---------- training ----------
    def training_step(self, batch, _):
        pose, tok = batch
        logits = self(pose, tok)
        loss = self.loss_fn(logits.reshape(-1, VOCAB_SIZE), tok[:, 1:].reshape(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ------------------------------------------------------------------
#  Convenience -----------------------------------------------------
def collate_fn(batch):
    pose, tok = zip(*batch)
    return (torch.nn.utils.rnn.pad_sequence(pose, batch_first=True), torch.stack(tok))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq_len", type=int, default=512)
    args = p.parse_args()

    ds = ChoreoGrooveDataset(args.data_root, args.seq_len)
    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    model = Choreo2GrooveModel(in_feats=102, lr=args.lr)
    pl.Trainer(max_epochs=args.epochs, accelerator="auto").fit(model, dl)


if __name__ == "__main__":
    main()
