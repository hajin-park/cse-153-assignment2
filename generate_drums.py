import torch, pretty_midi, argparse, os, numpy as np, subprocess
from choreo2groove import (
    Choreo2GrooveModel,
    DRUM_TOKENS,
    IDX2TOKEN,
    VOCAB_SIZE,
    BOS_IDX,
    EOS_IDX,
    SHIFT_SEC,
)


def tokens_to_midi(tokens):
    """Convert token list to PrettyMIDI, ignoring control tokens."""
    pm_obj, drums = pretty_midi.PrettyMIDI(), pretty_midi.Instrument(0, True)
    t = 0.0
    for tok in tokens:
        name = IDX2TOKEN[tok]
        if name in ("pad", "bos", "eos"):
            continue
        if name.startswith("shift_"):
            t += int(name.split("_")[1]) * SHIFT_SEC
            continue
        pitch = {
            "kick": 36,
            "snare": 38,
            "hihat_closed": 42,
            "hihat_open": 46,
            "tom_low": 45,
            "tom_mid": 47,
            "tom_high": 50,
            "crash": 49,
            "ride": 51,
        }[name]
        drums.notes.append(pretty_midi.Note(100, pitch, t, t + 0.1))
    pm_obj.instruments.append(drums)
    return pm_obj


def generate_drums(checkpoint_path, output_dir):
    """Generate drums using the trained model with CUDA support."""
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model - use VOCAB_SIZE from the original module
    model = Choreo2GrooveModel.load_from_checkpoint(
        checkpoint_path,
        in_feats=102,  # 17 joints * 3 coords * 2 (pos+vel)
        vocab_size=VOCAB_SIZE,  # Use the correct VOCAB_SIZE (111)
    )
    model.eval()
    model = model.to(device)

    pose_seq = torch.randn(1, 256, 102, device=device)  # dummy pose
    pose_duration = pose_seq.size(1) * SHIFT_SEC  # <<< DURATION

    memory = model.encoder(pose_seq)

    seq, elapsed = [BOS_IDX], 0.0  # <<< seed with BOS
    with torch.no_grad():
        for _ in range(255):
            tokens = torch.tensor([seq], device=device).long()
            logits = model(pose_seq, tokens)[0, -1]
            probs = torch.softmax(logits / 0.8, -1)
            nxt = torch.multinomial(probs, 1).item()
            seq.append(nxt)

            if IDX2TOKEN[nxt].startswith("shift_"):
                elapsed += int(IDX2TOKEN[nxt].split("_")[1]) * SHIFT_SEC
            if nxt == EOS_IDX or elapsed >= pose_duration:
                break  # <<< stop on sync

    midi_path = os.path.join(output_dir, "generated_drums.mid")
    tokens_to_midi(seq).write(midi_path)
    print(f"Generated drums saved to {midi_path}")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save outputs"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_drums(args.checkpoint, args.output_dir)
