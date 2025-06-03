import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import pretty_midi as pm
import torch
import json
from datetime import datetime
import seaborn as sns

# Import from existing files
import sys
sys.path.append('.')
from choreo2groove import Choreo2GrooveModel, ChoreoGrooveDataset, DRUM_TOKENS, VOCAB_SIZE, IDX2TOKEN

def load_trained_model(checkpoint_path):
    """Load the trained model"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = Choreo2GrooveModel(in_feats=102, lr=1e-4)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def calculate_movement_energy(pose_data, window_size=5):
    """Calculate movement energy over time from pose data"""
    if len(pose_data.shape) == 3:
        pose_data = pose_data.reshape(pose_data.shape[0], -1)
    
    # Calculate velocity (frame-to-frame differences)
    velocities = np.diff(pose_data, axis=0)
    
    # Calculate energy as magnitude of velocity
    energy = np.sqrt(np.sum(velocities**2, axis=1))
    
    # Smooth with moving average
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        energy = np.convolve(energy, kernel, mode='same')
    
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
        elif token_name in ["kick", "snare", "hihat_closed", "hihat_open", "tom_low", "tom_mid", "tom_high", "crash", "ride"]:
            drum_events.append({
                'time': current_time,
                'type': token_name,
                'is_kick': token_name == "kick",
                'is_snare': token_name == "snare",
                'is_accent': token_name in ["kick", "snare", "crash"]
            })
    
    return drum_events

def calculate_movement_beat_correlation(pose_data, drum_events, fps=60):
    """Calculate correlation between movement energy and drum beats"""
    # Calculate movement energy
    energy = calculate_movement_energy(pose_data)
    
    # Create time axis for pose data
    pose_times = np.arange(len(energy)) / fps
    
    # Create drum intensity timeline
    max_time = pose_times[-1] if len(pose_times) > 0 else 10.0
    drum_timeline = np.zeros(len(pose_times))
    
    for event in drum_events:
        if event['time'] <= max_time:
            # Find closest frame
            frame_idx = int(event['time'] * fps)
            if frame_idx < len(drum_timeline):
                # Weight different drum types
                weight = 3.0 if event['is_accent'] else 1.0
                drum_timeline[frame_idx] += weight
    
    # Calculate correlation (align lengths)
    min_len = min(len(energy), len(drum_timeline))
    if min_len > 10:  # Need enough data points
        correlation = np.corrcoef(energy[:min_len], drum_timeline[:min_len])[0, 1]
    else:
        correlation = 0.0
    
    return correlation, energy, drum_timeline, pose_times

def analyze_beat_alignment(sample_idx=58):
    """Comprehensive analysis of beat-movement alignment"""
    print("🔍 CHOREO2GROOVE ALIGNMENT ANALYSIS")
    print("=" * 50)
    
    # Load model and data
    checkpoint_path = "lightning_logs/version_6/checkpoints"
    checkpoint_files = list(Path(checkpoint_path).glob("*.ckpt"))
    if not checkpoint_files:
        print("❌ No checkpoint found!")
        return
    
    model = load_trained_model(checkpoint_files[0])
    dataset = ChoreoGrooveDataset("dataset_root", seq_len=256)
    
    if sample_idx >= len(dataset):
        sample_idx = 0
    
    pose_data, original_tokens = dataset[sample_idx]
    
    print(f"🕺 Analyzing dance sample {sample_idx}")
    print(f"Pose data shape: {pose_data.shape}")
    
    # Generate new drum beat
    print("🎵 Generating AI drum beat...")
    pose_tensor = torch.from_numpy(pose_data.numpy()).unsqueeze(0)
    generated_tokens = []
    
    with torch.no_grad():
        memory = model.encoder(pose_tensor)
        current_seq = torch.tensor([DRUM_TOKENS["pad"]], dtype=torch.long).unsqueeze(0)
        
        for i in range(255):
            logits = model.decoder(current_seq, memory)
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated_tokens.append(next_token)
            current_seq = torch.cat([current_seq, torch.tensor([[next_token]])], dim=1)
            
            if len(generated_tokens) > 10 and all(t == DRUM_TOKENS["pad"] for t in generated_tokens[-5:]):
                break
    
    # Analyze alignments
    print("📊 Calculating alignment metrics...")
    
    # Extract drum events
    generated_drums = extract_drum_timing(generated_tokens)
    original_drums = extract_drum_timing(original_tokens.numpy())
    
    print(f"Generated drums: {len(generated_drums)} events")
    print(f"Original drums: {len(original_drums)} events")
    
    # Calculate correlations
    gen_corr, gen_energy, gen_timeline, times = calculate_movement_beat_correlation(
        pose_data.numpy(), generated_drums)
    orig_corr, _, orig_timeline, _ = calculate_movement_beat_correlation(
        pose_data.numpy(), original_drums)
    
    # Create analysis report
    analysis = {
        "sample_info": {
            "sample_id": sample_idx,
            "pose_frames": len(pose_data),
            "duration_seconds": len(pose_data) / 60.0,
        },
        "generated_analysis": {
            "drum_events": len(generated_drums),
            "movement_correlation": float(gen_corr) if not np.isnan(gen_corr) else 0.0,
            "kick_events": sum(1 for d in generated_drums if d['is_kick']),
            "snare_events": sum(1 for d in generated_drums if d['is_snare']),
            "total_accents": sum(1 for d in generated_drums if d['is_accent']),
        },
        "original_analysis": {
            "drum_events": len(original_drums),
            "movement_correlation": float(orig_corr) if not np.isnan(orig_corr) else 0.0,
            "kick_events": sum(1 for d in original_drums if d['is_kick']),
            "snare_events": sum(1 for d in original_drums if d['is_snare']),
            "total_accents": sum(1 for d in original_drums if d['is_accent']),
        },
        "alignment_quality": {
            "generated_vs_movement": "Good" if gen_corr > 0.2 else "Moderate" if gen_corr > 0.1 else "Weak",
            "compared_to_original": "Better" if gen_corr > orig_corr else "Similar" if abs(gen_corr - orig_corr) < 0.05 else "Worse",
            "correlation_difference": float(gen_corr - orig_corr) if not np.isnan(gen_corr - orig_corr) else 0.0
        }
    }
    
    print(f"🎯 ALIGNMENT RESULTS:")
    print(f"Generated correlation: {gen_corr:.3f}")
    print(f"Original correlation:  {orig_corr:.3f}")
    print(f"Quality: {analysis['alignment_quality']['generated_vs_movement']}")
    
    # Create visualization
    create_alignment_visualization(times, gen_energy, gen_timeline, orig_timeline, 
                                 generated_drums, original_drums, analysis, sample_idx)
    
    return analysis

def create_alignment_visualization(times, energy, gen_drums, orig_drums, 
                                 gen_events, orig_events, analysis, sample_idx):
    """Create comprehensive visualization of movement-beat alignment"""
    
    output_dir = Path("lightning_logs/version_6/complete_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create the alignment analysis plot
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'Choreo2Groove Alignment Analysis - Sample {sample_idx}', fontsize=16, fontweight='bold')
    
    # 1. Movement Energy
    axes[0].plot(times[:len(energy)], energy, 'b-', linewidth=2, label='Movement Energy')
    axes[0].set_title('Dance Movement Energy Over Time', fontweight='bold')
    axes[0].set_ylabel('Energy')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. Generated Drums with Movement
    axes[1].plot(times[:len(energy)], energy, 'b-', alpha=0.5, label='Movement Energy')
    axes[1].bar(times[:len(gen_drums)], gen_drums[:len(times)], alpha=0.7, color='red', 
               width=0.01, label='AI Generated Drums')
    
    # Mark specific drum types
    for event in gen_events:
        if event['time'] <= times[-1]:
            color = 'darkred' if event['is_kick'] else 'orange' if event['is_snare'] else 'pink'
            axes[1].axvline(x=event['time'], color=color, alpha=0.8, linewidth=2)
    
    axes[1].set_title(f"AI Generated Drums vs Movement (Correlation: {analysis['generated_analysis']['movement_correlation']:.3f})", 
                     fontweight='bold')
    axes[1].set_ylabel('Intensity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Original Drums with Movement
    axes[2].plot(times[:len(energy)], energy, 'b-', alpha=0.5, label='Movement Energy')
    axes[2].bar(times[:len(orig_drums)], orig_drums[:len(times)], alpha=0.7, color='green', 
               width=0.01, label='Original Drums')
    
    # Mark specific drum types
    for event in orig_events:
        if event['time'] <= times[-1]:
            color = 'darkgreen' if event['is_kick'] else 'lightgreen' if event['is_snare'] else 'lime'
            axes[2].axvline(x=event['time'], color=color, alpha=0.8, linewidth=2)
    
    axes[2].set_title(f"Original Drums vs Movement (Correlation: {analysis['original_analysis']['movement_correlation']:.3f})", 
                     fontweight='bold')
    axes[2].set_ylabel('Intensity')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Comparison Bar Chart
    categories = ['Drum Events', 'Kick Hits', 'Snare Hits', 'Total Accents']
    generated_values = [
        analysis['generated_analysis']['drum_events'],
        analysis['generated_analysis']['kick_events'],
        analysis['generated_analysis']['snare_events'],
        analysis['generated_analysis']['total_accents']
    ]
    original_values = [
        analysis['original_analysis']['drum_events'],
        analysis['original_analysis']['kick_events'],
        analysis['original_analysis']['snare_events'],
        analysis['original_analysis']['total_accents']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[3].bar(x - width/2, generated_values, width, label='AI Generated', color='red', alpha=0.7)
    axes[3].bar(x + width/2, original_values, width, label='Original', color='green', alpha=0.7)
    
    axes[3].set_title('Drum Pattern Comparison', fontweight='bold')
    axes[3].set_ylabel('Count')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(categories)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the analysis plot
    plot_path = output_dir / f"alignment_analysis_sample_{sample_idx}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save analysis data
    analysis_path = output_dir / f"alignment_report_sample_{sample_idx}.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Create summary text report
    create_alignment_summary(analysis, output_dir, sample_idx)
    
    print(f"📊 Alignment analysis saved:")
    print(f"  📈 Plot: {plot_path}")
    print(f"  📋 Report: {analysis_path}")
    
    return plot_path

def create_alignment_summary(analysis, output_dir, sample_idx):
    """Create human-readable alignment summary"""
    
    summary = f"""
# CHOREO2GROOVE ALIGNMENT ANALYSIS REPORT
## Sample {sample_idx} - Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 🎯 MAIN QUESTION: Are the drums aligned with the dance?

### 📊 CORRELATION ANALYSIS
- **AI Generated Drums ↔ Movement**: {analysis['generated_analysis']['movement_correlation']:.3f}
- **Original Drums ↔ Movement**: {analysis['original_analysis']['movement_correlation']:.3f}
- **Difference**: {analysis['alignment_quality']['correlation_difference']:+.3f}

### 🔍 WHAT THIS MEANS:
- **Correlation > 0.3**: Strong alignment (drums follow dance energy)
- **Correlation 0.1-0.3**: Moderate alignment (some relationship)
- **Correlation < 0.1**: Weak alignment (mostly random)

**Your AI's Performance**: {analysis['alignment_quality']['generated_vs_movement']}
**Compared to Original**: {analysis['alignment_quality']['compared_to_original']}

### 🥁 DRUM PATTERN ANALYSIS
```
                    AI Generated  │  Original
Drum Events:        {analysis['generated_analysis']['drum_events']:>8}      │  {analysis['original_analysis']['drum_events']:>8}
Kick Hits:          {analysis['generated_analysis']['kick_events']:>8}      │  {analysis['original_analysis']['kick_events']:>8}
Snare Hits:         {analysis['generated_analysis']['snare_events']:>8}      │  {analysis['original_analysis']['snare_events']:>8}
Total Accents:      {analysis['generated_analysis']['total_accents']:>8}      │  {analysis['original_analysis']['total_accents']:>8}
```

### 🎬 HOW TO VERIFY ALIGNMENT:

1. **Listen + Watch Together**:
   - Play `generated_drums.mid` 
   - Watch `dance_video_FIXED.gif`
   - Notice: Do drum hits match energetic movements?

2. **Look for Patterns**:
   - **Big movements** → Should trigger kick/snare
   - **Fast movements** → Should trigger hi-hats
   - **Pauses** → Should have fewer drum hits

3. **Compare to Random**:
   - Original correlation: {analysis['original_analysis']['movement_correlation']:.3f}
   - If AI correlation > 0.1, it's learning patterns
   - If AI correlation > original, it's working well!

### ✅ EVIDENCE OF ALIGNMENT:
"""

    if analysis['generated_analysis']['movement_correlation'] > 0.2:
        summary += "- ✅ STRONG: High correlation shows drums follow dance energy\n"
    elif analysis['generated_analysis']['movement_correlation'] > 0.1:
        summary += "- ✅ MODERATE: Some correlation shows basic pattern learning\n"
    else:
        summary += "- ⚠️ WEAK: Low correlation suggests more training needed\n"

    if analysis['generated_analysis']['movement_correlation'] > analysis['original_analysis']['movement_correlation']:
        summary += "- ✅ AI performs BETTER than original training data\n"
    
    if analysis['generated_analysis']['drum_events'] > 0:
        summary += f"- ✅ Generated {analysis['generated_analysis']['drum_events']} drum events (not silent)\n"
    
    if analysis['generated_analysis']['total_accents'] > 0:
        summary += f"- ✅ Used {analysis['generated_analysis']['total_accents']} accent hits (kicks/snares)\n"

    summary += f"""
### 🎵 NEXT STEPS:
1. Open `alignment_analysis_sample_{sample_idx}.png` to see visual proof
2. Play the MIDI files while watching the GIF
3. Look for drum hits during high-energy dance moments
4. Compare AI vs Original patterns

**Bottom Line**: {analysis['alignment_quality']['generated_vs_movement']} alignment detected!
Your AI {'IS' if analysis['generated_analysis']['movement_correlation'] > 0.15 else 'may be'} learning dance-to-drum relationships.
"""

    summary_path = output_dir / f"ALIGNMENT_SUMMARY_sample_{sample_idx}.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"📋 Summary: {summary_path}")

if __name__ == "__main__":
    print("🔍 Starting Choreo2Groove Alignment Analysis...")
    
    # Run analysis on the same sample used in demo
    analysis_results = analyze_beat_alignment(sample_idx=58)
    
    if analysis_results:
        print("\n" + "="*60)
        print("🎉 ALIGNMENT ANALYSIS COMPLETE!")
        print("\n📊 Quick Summary:")
        print(f"   Movement-Drum Correlation: {analysis_results['generated_analysis']['movement_correlation']:.3f}")
        print(f"   Quality Rating: {analysis_results['alignment_quality']['generated_vs_movement']}")
        print(f"   Compared to Original: {analysis_results['alignment_quality']['compared_to_original']}")
        print("\n📁 Check lightning_logs/version_6/complete_output/ for:")
        print("   - alignment_analysis_sample_58.png (visual proof)")
        print("   - ALIGNMENT_SUMMARY_sample_58.md (detailed report)")
        print("   - alignment_report_sample_58.json (raw data)") 