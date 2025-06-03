import os
import random
import shutil
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pretty_midi as pm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

output_dir = Path(f"lightning_logs/lightning_logs/version_0/complete_output")
output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir}")

SHIFT_SEC = 0.02  # beat time resolution
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

# sequence control tokens
DRUM_TOKENS["bos"] = len(DRUM_TOKENS)  # begin‑of‑sequence
DRUM_TOKENS["eos"] = len(DRUM_TOKENS)  # end‑of‑sequence

VOCAB_SIZE = len(DRUM_TOKENS)
IDX2TOKEN = {v: k for k, v in DRUM_TOKENS.items()}
PAD_IDX = DRUM_TOKENS["pad"]
BOS_IDX = DRUM_TOKENS["bos"]
EOS_IDX = DRUM_TOKENS["eos"]


def _pitch_to_token(p: int) -> str:
    # General MIDI -> symbolic token
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


def midi_to_tokens(mid: pm.PrettyMIDI, time_unit: float = SHIFT_SEC) -> List[int]:
    """Drum MIDI -> event tokens (no BOS/EOS)."""
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


def collate_fn(batch):
    pose, tok = zip(*batch)
    return (torch.nn.utils.rnn.pad_sequence(pose, batch_first=True), torch.stack(tok))


class ChoreoGrooveDataset(Dataset):
    def __init__(self, root: str, seq_len: int = 512):
        self.items = sorted(glob(os.path.join(root, "*", "pose.npy")))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        pose_path = self.items[idx]
        drum_path = pose_path.replace("pose.npy", "drums.mid")

        # pose -> features
        pose = np.load(pose_path).reshape(-1, 51)  # (T, 17x3)
        vel = np.diff(pose, axis=0, prepend=pose[:1])
        feats = np.concatenate([pose, vel], axis=-1)  # (T, 102)
        feats = (feats - feats.mean()) / (feats.std() + 1e-5)
        feats = feats.astype(np.float32)

        # drums -> tokens
        tokens = [BOS_IDX] + midi_to_tokens(pm.PrettyMIDI(drum_path)) + [EOS_IDX]
        if len(tokens) < self.seq_len:
            tokens += [PAD_IDX] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[: self.seq_len]

        return torch.from_numpy(feats), torch.tensor(tokens, dtype=torch.long)


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

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.gru(x)
        return self.proj(x).transpose(0, 1)


class DrumDecoder(nn.Module):
    def __init__(self, embed=256, layers=4, nhead=8, vocab=VOCAB_SIZE):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, embed)
        self.pos_emb = nn.Embedding(1024, embed)
        dec_layer = nn.TransformerDecoderLayer(embed, nhead, 1024, batch_first=True)
        self.transformer = nn.TransformerDecoder(dec_layer, layers)
        self.fc_out = nn.Linear(embed, vocab)

    def forward(self, tgt, memory):
        pos = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        tgt = self.tok_emb(tgt) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )
        out = self.transformer(tgt, memory.transpose(0, 1), tgt_mask=mask)
        return self.fc_out(out)


class Choreo2GrooveModel(pl.LightningModule):
    def __init__(self, in_feats: int, lr=1e-4):
        super().__init__()
        self.encoder = PoseEncoder(in_feats)
        self.decoder = DrumDecoder()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.save_hyperparameters()

    # forward
    def forward(self, poses, tokens):
        memory = self.encoder(poses)
        if self.training:
            tgt_in = tokens[:, :-1]  # strip last (EOS / PAD)
            return self.decoder(tgt_in, memory)
        else:
            return self.decoder(tokens, memory)

    # training
    def training_step(self, batch, _):
        pose, tok = batch
        logits = self(pose, tok)
        loss = self.loss_fn(logits.reshape(-1, VOCAB_SIZE), tok[:, 1:].reshape(-1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


epochs = 30
lr = 1e-4
batch_size = 4
seq_len = 256
version = 0


def check_gpu_availability():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Using GPU")
        return True, gpu_count
    else:
        print("Using CPU")
        return False, 0


# Check GPU availability
has_gpu, gpu_count = check_gpu_availability()


# Setup DataLoader - optimized for GPU
num_workers = (
    0 if sys.platform.startswith("win") else min(4, gpu_count * 2) if has_gpu else 2
)
pin_memory = has_gpu  # Use pinned memory for faster GPU transfer


# Adjust batch size for GPU if available
if has_gpu and batch_size < 8:
    original_batch_size = batch_size
    batch_size = min(16, batch_size * 2)  # Increase batch size for GPU
    print(
        f"GPU detected: increasing batch size from {original_batch_size} to {batch_size}"
    )

dataset = ChoreoGrooveDataset("dataset_root", seq_len=seq_len)


# Calculate input features from first sample
sample_pose, _ = dataset[0]
in_feats = sample_pose.shape[-1]
print(f"Dataset loaded: {len(dataset)} samples, {in_feats} features per frame")

model = Choreo2GrooveModel(in_feats=in_feats, lr=lr)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=num_workers > 0,
)

checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    filename="choreo2groove-{epoch:02d}-{train_loss:.3f}",
    save_top_k=1,
    mode="min",
    save_last=True,
)

logger = TensorBoardLogger("lightning_logs", version=version)

trainer_kwargs = {
    "max_epochs": epochs,
    "callbacks": [checkpoint_callback],
    "logger": logger,
    "log_every_n_steps": 10,
    "check_val_every_n_epoch": 1,
    "enable_progress_bar": True,
    "enable_model_summary": True,
}


if has_gpu:
    trainer_kwargs.update(
        {
            "accelerator": "gpu",
            "devices": min(gpu_count, 1),  # Use 1 GPU for now
            "precision": "16-mixed",  # Mixed precision for faster training
        }
    )
    print("GPU training enabled with mixed precision")
else:
    trainer_kwargs.update(
        {
            "accelerator": "cpu",
            "devices": 1,
        }
    )
    print("CPU training mode")

trainer = pl.Trainer(**trainer_kwargs)

# Start training
print(f"Starting training for {epochs} epochs...")
start_time = datetime.now()


trainer.fit(model, dataloader)

end_time = datetime.now()
training_duration = end_time - start_time
print(f"\nTraining completed")
print(f"Training duration: {training_duration}")

# Get final metrics
final_loss = trainer.callback_metrics.get("train_loss", "unknown")
print(f"Final training loss: {final_loss}")

sys.path.append(".")


def load_trained_model(checkpoint_path):
    """Load the trained model with GPU support"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Choreo2GrooveModel(in_feats=102, lr=1e-4)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model = model.to(device)
    return model


def token_to_pitch(token_name):
    """Convert token name back to MIDI pitch"""
    pitch_map = {
        "kick": 36,
        "snare": 38,
        "hihat_closed": 42,
        "hihat_open": 46,
        "tom_low": 45,
        "tom_mid": 47,
        "tom_high": 50,
        "crash": 49,
        "ride": 51,
    }
    return pitch_map.get(token_name, 38)


def tokens_to_midi(tokens, time_unit=SHIFT_SEC, bpm=120):
    """Convert drum tokens back to MIDI"""
    midi = pm.PrettyMIDI(initial_tempo=bpm)
    drums = pm.Instrument(program=0, is_drum=True, name="Generated_Drums")

    current_time = 0.0

    for token_id in tokens:
        if token_id >= VOCAB_SIZE:
            continue

        token_name = IDX2TOKEN.get(token_id, "unknown")
        if token_name in ("pad", "bos", "eos"):  # skip BOS/EOS
            continue
        elif token_name.startswith("shift_"):
            shift_amount = int(token_name.split("_")[1])
            current_time += shift_amount * time_unit
        elif token_name in [
            "kick",
            "snare",
            "hihat_closed",
            "hihat_open",
            "tom_low",
            "tom_mid",
            "tom_high",
            "crash",
            "ride",
        ]:
            pitch = token_to_pitch(token_name)
            velocity = random.randint(80, 120)
            note = pm.Note(pitch, velocity, current_time, current_time + 0.1)
            drums.notes.append(note)

    midi.instruments.append(drums)
    return midi


def generate_drum_beat(model, pose_data, max_length=256):
    device = next(model.parameters()).device
    pose_tensor = torch.from_numpy(pose_data).unsqueeze(0).to(device)

    memory = model.encoder(pose_tensor)
    pose_dur = pose_data.shape[0] * SHIFT_SEC  # duration

    seq, elapsed = [BOS_IDX], 0.0
    with torch.no_grad():
        for _ in range(max_length):
            cur = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            logits = model.decoder(cur, memory)[0, -1]
            nxt = torch.multinomial(torch.softmax(logits / 0.8, -1), 1).item()
            seq.append(nxt)

            if IDX2TOKEN[nxt].startswith("shift_"):
                elapsed += int(IDX2TOKEN[nxt].split("_")[1]) * SHIFT_SEC
            if nxt == EOS_IDX or elapsed >= pose_dur:
                break
    return seq


# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

latest_checkpoint = "lightning_logs/lightning_logs/version_0/checkpoints/last.ckpt"
output_dir = Path(f"lightning_logs/lightning_logs/version_{version}/complete_output")

output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir}")


model = load_trained_model(latest_checkpoint)
dataset = ChoreoGrooveDataset("dataset_root", seq_len=256)
sample_idx = 58  # arbitrary test

pose_data, original_tokens = dataset[sample_idx]

# Get sample metadata
metadata_path = Path(f"dataset_root/sample_{sample_idx:03d}/metadata.txt")
sample_info = {}
if metadata_path.exists():
    with open(metadata_path) as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                sample_info[key] = value.strip()
    # Copy metadata
    shutil.copy(metadata_path, output_dir / "dance_metadata.txt")

generated_tokens = generate_drum_beat(model, pose_data.numpy())
generated_midi = tokens_to_midi(generated_tokens)
original_midi = tokens_to_midi(original_tokens.numpy())

generated_midi.write(str(output_dir / "generated_drums.mid"))
original_midi.write(str(output_dir / "original_drums.mid"))

output_dir.mkdir(exist_ok=True)


def create_pose_visualization(pose_data, output_path, fps=30):
    """Create a video visualization of the pose data"""
    print("Creating dance visualization...")

    # COCO-17 skeleton connections
    skeleton = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],  # head
        [5, 11],
        [6, 12],
        [5, 6],  # torso
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],  # arms
        [11, 13],
        [12, 14],
        [13, 15],
        [14, 16],  # legs
    ]

    # Normalize pose data to [-1, 1] range
    pose_flat = pose_data.reshape(len(pose_data), -1, 3)

    # Calculate bounds for normalization
    all_coords = pose_flat.reshape(-1, 3)
    x_min, x_max = np.percentile(all_coords[:, 0], [5, 95])
    y_min, y_max = np.percentile(all_coords[:, 1], [5, 95])

    # Normalize coordinates
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    scale = max(x_max - x_min, y_max - y_min) / 1.8

    pose_normalized = pose_flat.copy()
    pose_normalized[:, :, 0] = (pose_normalized[:, :, 0] - x_center) / scale
    pose_normalized[:, :, 1] = (pose_normalized[:, :, 1] - y_center) / scale

    # Subsample frames for reasonable video length
    stride = max(1, len(pose_normalized) // (fps * 10))
    pose_frames = pose_normalized[::stride]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_facecolor("black")

    def animate(frame_idx):
        ax.clear()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_title(
            f"Choreo2Groove - Dance Visualization (Frame {frame_idx+1}/{len(pose_frames)})",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        ax.set_facecolor("black")

        if frame_idx < len(pose_frames):
            frame = pose_frames[frame_idx]

            # Draw skeleton connections
            for connection in skeleton:
                if connection[0] < len(frame) and connection[1] < len(frame):
                    x_coords = [frame[connection[0]][0], frame[connection[1]][0]]
                    y_coords = [frame[connection[0]][1], frame[connection[1]][1]]
                    ax.plot(x_coords, y_coords, "c-", linewidth=2, alpha=0.8)

            # Draw joints
            for i, joint in enumerate(frame):
                color = "red" if i in [0, 1, 2, 3, 4] else "yellow"
                ax.scatter(joint[0], joint[1], c=color, s=50, alpha=0.9)

        ax.text(
            0.02,
            0.98,
            "Generated by Choreo2Groove AI",
            transform=ax.transAxes,
            fontsize=10,
            color="lime",
            weight="bold",
            va="top",
        )
        ax.text(
            0.02,
            0.02,
            f"Dance Style: Basic Moves | Duration: {len(pose_data)/60:.1f}s",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            va="bottom",
        )

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(pose_frames),
        interval=1000 // fps,
        blit=False,
        repeat=True,
    )

    try:
        anim.save(str(output_path), writer="pillow", fps=fps)
        print(f"Dance video created: {output_path}")
    except Exception as e:
        print(f"Video creation failed: {e}")
        gif_path = output_path.with_suffix(".gif")
        anim.save(str(gif_path), writer="pillow", fps=fps // 2)
        print(f"Created GIF instead: {gif_path}")
        return gif_path

    plt.close(fig)
    return output_path


raw_pose_path = Path(f"dataset_root/sample_{sample_idx:03d}/pose.npy")
raw_pose_data = np.load(raw_pose_path)
video_path = create_pose_visualization(raw_pose_data, output_dir / "pose.gif")


def calculate_movement_energy(pose_data, window_size=5):
    """Calculate movement energy over time from pose data"""
    if len(pose_data.shape) == 3:
        pose_data = pose_data.reshape(pose_data.shape[0], -1)

    velocities = np.diff(pose_data, axis=0)
    energy = np.sqrt(np.sum(velocities**2, axis=1))

    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        energy = np.convolve(energy, kernel, mode="same")

    return energy


def extract_drum_timing(tokens, time_unit=0.02):
    """Extract drum hit timings and types from token sequence"""
    drum_events = []
    current_time = 0.0

    for token_id in tokens:
        if token_id >= VOCAB_SIZE:
            continue

        token_name = IDX2TOKEN.get(token_id, "unknown")

        if token_name.startswith("shift_"):
            shift_amount = int(token_name.split("_")[1])
            current_time += shift_amount * time_unit
        elif token_name in [
            "kick",
            "snare",
            "hihat_closed",
            "hihat_open",
            "tom_low",
            "tom_mid",
            "tom_high",
            "crash",
            "ride",
        ]:
            drum_events.append(
                {
                    "time": current_time,
                    "type": token_name,
                    "is_kick": token_name == "kick",
                    "is_snare": token_name == "snare",
                    "is_accent": token_name in ["kick", "snare", "crash"],
                }
            )

    return drum_events


def calculate_movement_beat_correlation(pose_data, drum_events, fps=60):
    """Calculate correlation between movement energy and drum beats"""
    energy = calculate_movement_energy(pose_data)
    pose_times = np.arange(len(energy)) / fps
    max_time = pose_times[-1] if len(pose_times) > 0 else 10.0
    drum_timeline = np.zeros(len(pose_times))

    for event in drum_events:
        if event["time"] <= max_time:
            frame_idx = int(event["time"] * fps)
            if frame_idx < len(drum_timeline):
                weight = 3.0 if event["is_accent"] else 1.0
                drum_timeline[frame_idx] += weight

    min_len = min(len(energy), len(drum_timeline))
    if min_len > 10:
        correlation = np.corrcoef(energy[:min_len], drum_timeline[:min_len])[0, 1]
    else:
        correlation = 0.0

    return correlation, energy, drum_timeline, pose_times


def create_alignment_visualization(
    times,
    energy,
    gen_drums,
    orig_drums,
    gen_events,
    orig_events,
    analysis,
    sample_idx,
    output_dir,
):
    """Create comprehensive visualization of movement-beat alignment"""

    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(
        f"Choreo2Groove Alignment Analysis - Sample {sample_idx}",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Movement Energy
    axes[0].plot(
        times[: len(energy)], energy, "b-", linewidth=2, label="Movement Energy"
    )
    axes[0].set_title("Dance Movement Energy Over Time", fontweight="bold")
    axes[0].set_ylabel("Energy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Generated Drums with Movement
    axes[1].plot(times[: len(energy)], energy, "b-", alpha=0.5, label="Movement Energy")
    axes[1].bar(
        times[: len(gen_drums)],
        gen_drums[: len(times)],
        alpha=0.7,
        color="red",
        width=0.01,
        label="AI Generated Drums",
    )

    for event in gen_events:
        if event["time"] <= times[-1]:
            color = (
                "darkred"
                if event["is_kick"]
                else "orange" if event["is_snare"] else "pink"
            )
            axes[1].axvline(x=event["time"], color=color, alpha=0.8, linewidth=2)

    axes[1].set_title(
        f"AI Generated Drums vs Movement (Correlation: {analysis['generated_analysis']['movement_correlation']:.3f})",
        fontweight="bold",
    )
    axes[1].set_ylabel("Intensity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Original Drums with Movement
    axes[2].plot(times[: len(energy)], energy, "b-", alpha=0.5, label="Movement Energy")
    axes[2].bar(
        times[: len(orig_drums)],
        orig_drums[: len(times)],
        alpha=0.7,
        color="green",
        width=0.01,
        label="Original Drums",
    )

    for event in orig_events:
        if event["time"] <= times[-1]:
            color = (
                "darkgreen"
                if event["is_kick"]
                else "lightgreen" if event["is_snare"] else "lime"
            )
            axes[2].axvline(x=event["time"], color=color, alpha=0.8, linewidth=2)

    axes[2].set_title(
        f"Original Drums vs Movement (Correlation: {analysis['original_analysis']['movement_correlation']:.3f})",
        fontweight="bold",
    )
    axes[2].set_ylabel("Intensity")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 4. Comparison Bar Chart
    categories = ["Drum Events", "Kick Hits", "Snare Hits", "Total Accents"]
    generated_values = [
        analysis["generated_analysis"]["drum_events"],
        analysis["generated_analysis"]["kick_events"],
        analysis["generated_analysis"]["snare_events"],
        analysis["generated_analysis"]["total_accents"],
    ]
    original_values = [
        analysis["original_analysis"]["drum_events"],
        analysis["original_analysis"]["kick_events"],
        analysis["original_analysis"]["snare_events"],
        analysis["original_analysis"]["total_accents"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    axes[3].bar(
        x - width / 2,
        generated_values,
        width,
        label="AI Generated",
        color="red",
        alpha=0.7,
    )
    axes[3].bar(
        x + width / 2,
        original_values,
        width,
        label="Original",
        color="green",
        alpha=0.7,
    )

    axes[3].set_title("Drum Pattern Comparison", fontweight="bold")
    axes[3].set_ylabel("Count")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(categories)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = output_dir / f"alignment_analysis_sample_{sample_idx}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


generated_drums = extract_drum_timing(generated_tokens)
original_drums = extract_drum_timing(original_tokens.numpy())

gen_corr, gen_energy, gen_timeline, times = calculate_movement_beat_correlation(
    pose_data.numpy(), generated_drums
)
orig_corr, _, orig_timeline, _ = calculate_movement_beat_correlation(
    pose_data.numpy(), original_drums
)

analysis = {
    "sample_info": {
        "sample_id": sample_idx,
        "pose_frames": len(pose_data),
        "duration_seconds": len(pose_data) / 60.0,
    },
    "generated_analysis": {
        "drum_events": len(generated_drums),
        "movement_correlation": float(gen_corr) if not np.isnan(gen_corr) else 0.0,
        "kick_events": sum(1 for d in generated_drums if d["is_kick"]),
        "snare_events": sum(1 for d in generated_drums if d["is_snare"]),
        "total_accents": sum(1 for d in generated_drums if d["is_accent"]),
    },
    "original_analysis": {
        "drum_events": len(original_drums),
        "movement_correlation": (float(orig_corr) if not np.isnan(orig_corr) else 0.0),
        "kick_events": sum(1 for d in original_drums if d["is_kick"]),
        "snare_events": sum(1 for d in original_drums if d["is_snare"]),
        "total_accents": sum(1 for d in original_drums if d["is_accent"]),
    },
    "alignment_quality": {
        "generated_vs_movement": (
            "Good" if gen_corr > 0.2 else "Moderate" if gen_corr > 0.1 else "Weak"
        ),
        "compared_to_original": (
            "Better"
            if gen_corr > orig_corr
            else "Similar" if abs(gen_corr - orig_corr) < 0.05 else "Worse"
        ),
        "correlation_difference": (
            float(gen_corr - orig_corr) if not np.isnan(gen_corr - orig_corr) else 0.0
        ),
    },
}

plot_path = create_alignment_visualization(
    times,
    gen_energy,
    gen_timeline,
    orig_timeline,
    generated_drums,
    original_drums,
    analysis,
    sample_idx,
    output_dir,
)

print(f"Image at: {plot_path}")
