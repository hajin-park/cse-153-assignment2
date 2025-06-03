import numpy as np
import pretty_midi
import json
import os

def create_random_midi(num_events=40, output_path='random_baseline.mid'):
    """Create a random MIDI file with drum events."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    
    # Random times between 0 and 4 seconds
    times = np.sort(np.random.uniform(0, 4, num_events))
    
    # Random drum notes (35-81 is the standard MIDI drum range)
    for time in times:
        note = np.random.randint(35, 82)
        note_on = pretty_midi.Note(velocity=100, pitch=note, start=time, end=time + 0.1)
        drums.notes.append(note_on)
    
    pm.instruments.append(drums)
    pm.write(output_path)
    
    # Create metadata
    metadata = {
        "num_events": num_events,
        "duration": 4.0,
        "note_range": "35-81",
        "description": "Random baseline drum pattern"
    }
    
    # Save metadata
    metadata_path = output_path.replace('.mid', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    os.makedirs("random_baselines", exist_ok=True)
    
    # Create multiple random baselines
    for i in range(5):
        output_path = f"random_baselines/random_baseline_{i}.mid"
        create_random_midi(output_path=output_path)
        print(f"Created random baseline {i}") 