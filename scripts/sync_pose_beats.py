#!/usr/bin/env python
"""Trim/quantise pose & MIDI so every 20 ms frame aligns to token grid."""
import argparse, pathlib, numpy as np, miditoolkit, tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root", required=True)
args = ap.parse_args()

clips = list(pathlib.Path(args.dataset_root).rglob("pose.npy"))
for pose_path in tqdm.tqdm(clips):
    pose = np.load(pose_path)
    target_len = (pose.shape[0] // 3) * 3  # multiple of 3 frames (≈60 fps)
    np.save(pose_path, pose[:target_len])

    midi_path = pose_path.with_name("drums.mid")
    midi = miditoolkit.MidiFile(str(midi_path))
    midi.tempo_changes[0].bpm = 60  # 1 beat = 1 s for easy 20 ms grid
    midi.dump(str(midi_path))
print("✓ Pose & MIDI synchronised")
