#!/usr/bin/env python
"""Run ADTLib CNN on WAV files and write drums.mid."""
import argparse, pathlib, tqdm, librosa, pretty_midi as pm
from adtlib.inference import OnsetsADTPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--audio_root", required=True)
ap.add_argument("--out_root", required=True)
args = ap.parse_args()

pipeline = OnsetsADTPipeline(model="CarlSouthall/adlib_cnn")

wav_paths = list(pathlib.Path(args.audio_root).rglob("*.wav"))
for wav in tqdm.tqdm(wav_paths):
    y, sr = librosa.load(wav, sr=22050, mono=True)
    preds = pipeline.process(y, sr)  # keys: kick/snare/hihat
    inst = pm.Instrument(program=0, is_drum=True)
    mapping = {"kick": 36, "snare": 38, "hihat": 42}
    for name, times in preds.items():
        for t in times:
            inst.notes.append(pm.Note(100, mapping[name], t, t + 0.05))
    m = pm.PrettyMIDI()
    m.instruments.append(inst)
    out_dir = pathlib.Path(args.out_root, wav.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    m.write(out_dir / "drums.mid")
print("✓ MIDI created for", len(wav_paths), "clips")
