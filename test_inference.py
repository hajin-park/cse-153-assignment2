#!/usr/bin/env python
"""
Test inference with the trained Choreo2Groove model
Generate drum beats and convert to audio for listening
"""
import torch
import numpy as np
import pretty_midi as pm
from pathlib import Path
import random

from choreo2groove import Choreo2GrooveModel, ChoreoGrooveDataset, DRUM_TOKENS, IDX2TOKEN, VOCAB_SIZE

def load_trained_model(checkpoint_path, in_feats=102):
    """Load the trained model from checkpoint"""
    print(f"🔄 Loading model from {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same architecture
    model = Choreo2GrooveModel(in_feats=in_feats, lr=1e-4)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model

def tokens_to_midi(tokens, time_unit=0.02, bpm=120):
    """Convert drum tokens back to MIDI"""
    midi = pm.PrettyMIDI(initial_tempo=bpm)
    drums = pm.Instrument(program=0, is_drum=True, name="Generated_Drums")
    
    current_time = 0.0
    
    for token_id in tokens:
        if token_id >= VOCAB_SIZE:
            continue
            
        token_name = IDX2TOKEN.get(token_id, "unknown")
        
        if token_name.startswith("shift_"):
            # Time shift token
            shift_amount = int(token_name.split("_")[1])
            current_time += shift_amount * time_unit
        elif token_name in ["kick", "snare", "hihat_closed", "hihat_open", "tom_low", "tom_mid", "tom_high", "crash", "ride"]:
            # Drum hit token
            pitch = token_to_pitch(token_name)
            velocity = random.randint(80, 120)  # Add some velocity variation
            note = pm.Note(pitch, velocity, current_time, current_time + 0.1)
            drums.notes.append(note)
    
    midi.instruments.append(drums)
    return midi

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
        "ride": 51
    }
    return pitch_map.get(token_name, 38)  # Default to snare

def generate_drum_beat(model, pose_data, max_length=256):
    """Generate drum beat from pose data using trained model"""
    print("🎵 Generating drum beat...")
    
    # Prepare pose data
    pose_tensor = torch.from_numpy(pose_data).unsqueeze(0)  # Add batch dimension
    
    # Start with padding token
    generated_tokens = [DRUM_TOKENS["pad"]]
    
    with torch.no_grad():
        # Encode the pose sequence
        memory = model.encoder(pose_tensor)  # (T, 1, embed_dim)
        
        # Generate tokens one by one
        for i in range(max_length - 1):
            # Current sequence as tensor
            current_seq = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
            
            # Get next token probabilities
            logits = model.decoder(current_seq, memory)  # (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1, :]  # Last token predictions
            
            # Sample next token (with some randomness for variety)
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)  # Temperature sampling
            next_token = torch.multinomial(probs, 1).item()
            
            generated_tokens.append(next_token)
            
            # Stop if we hit too many pad tokens in a row
            if len(generated_tokens) > 10 and all(t == DRUM_TOKENS["pad"] for t in generated_tokens[-5:]):
                break
    
    print(f"✅ Generated {len(generated_tokens)} drum tokens")
    return generated_tokens

def test_model_inference():
    """Test the trained model and generate audio output"""
    
    # Find the latest checkpoint
    checkpoint_dir = Path("lightning_logs/version_6/checkpoints")
    if not checkpoint_dir.exists():
        print("❌ No checkpoint found! Train the model first.")
        return
    
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        print("❌ No checkpoint files found!")
        return
    
    latest_checkpoint = checkpoint_files[0]  # Take the first (should be only one)
    print(f"📁 Using checkpoint: {latest_checkpoint}")
    
    # Load the trained model
    model = load_trained_model(latest_checkpoint)
    
    # Load a sample from the dataset
    dataset = ChoreoGrooveDataset("dataset_root", seq_len=256)
    if len(dataset) == 0:
        print("❌ No data found in dataset_root!")
        return
    
    # Pick a random sample
    sample_idx = random.randint(0, len(dataset) - 1)
    pose_data, original_tokens = dataset[sample_idx]
    
    print(f"🕺 Using dance sample {sample_idx}")
    print(f"Pose shape: {pose_data.shape}")
    print(f"Original tokens length: {len(original_tokens)}")
    
    # Generate new drum beat
    generated_tokens = generate_drum_beat(model, pose_data.numpy())
    
    # Convert to MIDI
    print("🎼 Converting to MIDI...")
    generated_midi = tokens_to_midi(generated_tokens)
    original_midi = tokens_to_midi(original_tokens.numpy())
    
    # Save outputs
    output_dir = Path("generated_output")
    output_dir.mkdir(exist_ok=True)
    
    generated_path = output_dir / f"generated_drums_sample_{sample_idx}.mid"
    original_path = output_dir / f"original_drums_sample_{sample_idx}.mid"
    
    generated_midi.write(str(generated_path))
    original_midi.write(str(original_path))
    
    print(f"💾 Saved generated drums: {generated_path}")
    print(f"💾 Saved original drums: {original_path}")
    
    # Try to convert to WAV for listening
    try:
        convert_to_audio(generated_path, output_dir / f"generated_drums_sample_{sample_idx}.wav")
        convert_to_audio(original_path, output_dir / f"original_drums_sample_{sample_idx}.wav")
        print("🔊 Audio files created! You can listen to them now.")
    except Exception as e:
        print(f"⚠️ Audio conversion failed: {e}")
        print("💡 You can still listen to the .mid files in a MIDI player or DAW")
    
    return generated_path, original_path

def convert_to_audio(midi_path, wav_path):
    """Convert MIDI to WAV using FluidSynth or similar"""
    try:
        # Try using pygame for simple playback
        import subprocess
        
        # Check if fluidsynth is available
        result = subprocess.run(['fluidsynth', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            # Use fluidsynth if available
            cmd = [
                'fluidsynth', '-ni', '-g', '0.5', '-F', str(wav_path),
                '/path/to/soundfont.sf2', str(midi_path)  # You'd need a soundfont
            ]
            subprocess.run(cmd, check=True)
        else:
            print("💡 FluidSynth not found. Install it for audio conversion:")
            print("   https://github.com/FluidSynth/fluidsynth")
            
    except ImportError:
        print("💡 For audio playback, install additional packages:")
        print("   pip install pygame fluidsynth")
    except Exception as e:
        print(f"⚠️ Audio conversion error: {e}")

if __name__ == "__main__":
    print("🎵 CHOREO2GROOVE INFERENCE TEST 🕺")
    print("=" * 50)
    
    generated_file, original_file = test_model_inference()
    
    print("\\n🎉 INFERENCE COMPLETE!")
    print("\\n📁 Your generated drum beats are ready:")
    print(f"  Generated: {generated_file}")
    print(f"  Original:  {original_file}")
    print("\\n🔊 How to listen:")
    print("  1. Double-click the .mid files to play in your default MIDI player")
    print("  2. Import into GarageBand, FL Studio, or any DAW")
    print("  3. Use online MIDI players like: https://onlinesequencer.net/import")
    
    print("\\n🎵 Compare the generated vs original drums to see how well your model learned!") 