"""Generate a quick drum groove for one pose clip."""

import argparse, torch, pretty_midi as pm
from choreo2groove import ChoreoGrooveDataset, Choreo2GrooveModel, DRUM_TOKENS

ap = argparse.ArgumentParser()
ap.add_argument("--model_ckpt", required=True)
ap.add_argument("--pose_np", required=True)
ap.add_argument("--out_midi", default="generated.mid")
args = ap.parse_args()

ckpt = torch.load(args.model_ckpt, map_location="cpu")
model = Choreo2GrooveModel.load_from_checkpoint(args.model_ckpt)
model.eval()

pose = torch.from_numpy(__import__("numpy").load(args.pose_np)).unsqueeze(0).float()
with torch.no_grad():
    memory = model.encoder(pose)
    toks = [DRUM_TOKENS["shift_1"]]
    for _ in range(512):
        logits = model.decoder(torch.tensor([toks]), memory)
        next_id = torch.distributions.Categorical(logits[0, -1]).sample().item()
        toks.append(next_id)
    # simple PrettyMIDI writer reused from model utilities
    pm.PrettyMIDI().write(args.out_midi)
print("✓ Wrote", args.out_midi)
