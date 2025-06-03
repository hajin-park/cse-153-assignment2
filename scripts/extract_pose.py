#!/usr/bin/env python
"""Convert AIST++ NPZ key‑points into pose.npy (T, J, 3) float32 arrays."""
import argparse, pathlib, numpy as np, tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--npz_root", required=True)
ap.add_argument("--out_root", required=True)
args = ap.parse_args()

npz_files = list(pathlib.Path(args.npz_root).rglob("*.npz"))
for f in tqdm.tqdm(npz_files):
    data = np.load(f, allow_pickle=True)
    xyz = data["coords3d"].astype(np.float32)  # (T, J, 3)
    out_dir = pathlib.Path(args.out_root, f.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pose.npy", xyz)
print("✓ pose.npy written for", len(npz_files), "clips")
