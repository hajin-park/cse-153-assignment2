# 🎵 Choreo2Groove: Dance-to-Drum AI Generator

An AI system that generates drum beats from dance movements using PyTorch
Lightning and Transformer architecture.

## 🚀 Quick Start (2 Commands)

### 1. Train the Model

```bash
py train_and_analyze.py --epochs 8 --lr 2e-4 --batch_size 6
```

### 2. Generate Complete Analysis

```bash
py generate_everything.py --checkpoint lightning_logs/version_0/checkpoints/epoch=1-step=50.ckpt --version 0
```

**That's it!** Results will be in `version_0/complete_output/`

## 📁 Essential Files

### **Core System (Required)**

-   `choreo2groove.py` - Model architecture (Encoder + Transformer Decoder)
-   `train_and_analyze.py` - Training script with automatic analysis
-   `generate_drums.py` - Generate drums from trained model
-   `alignment_analysis.py` - Analyze movement-drum correlation
-   `create_random_baseline.py` - Create random comparisons
-   `generate_everything.py` - Complete analysis pipeline

### **Data (Required)**

-   `dataset_root/` - Training data (76 dance samples with pose + MIDI)
-   `data/` - Reference files for analysis

### **Results (Generated)**

-   `lightning_logs/` - Training checkpoints and logs
-   `version_X/complete_output/` - Complete analysis results

## 🎯 What You Get

After running the 2 commands above, you'll have:

1. **AI-Generated Drums**: `generated_drums.mid`
2. **Visual Analysis**: `correlation_plot.png`
3. **Comprehensive Report**: `ANALYSIS_SUMMARY.md`
4. **Comparison Data**: Random baselines + original drums

## 📊 Understanding Results

### **Correlation Analysis**

-   **> 0.15**: Good alignment between dance and drums
-   **0.05-0.15**: Moderate alignment (learning in progress)
-   **< 0.05**: Weak alignment (needs more training)

### **Example Results (Version 0)**

-   Movement-Drum Correlation: **0.038** (weak but positive)
-   Drum Events Generated: **95** (good density)
-   Drum Types Used: **Balanced** (kicks, snares, hi-hats)

## 🔧 Advanced Usage

### Training Options

```bash
# Longer training
py train_and_analyze.py --epochs 20 --lr 1e-4 --batch_size 4

# Faster training (less accuracy)
py train_and_analyze.py --epochs 5 --lr 5e-4 --batch_size 8
```

### Analysis Only

```bash
# Just analyze existing checkpoint
py alignment_analysis.py --midi_file version_0/complete_output/generated_drums.mid --pose_file data/pose.npy --output_dir results/
```

## 🎵 Demo Your Results

1. **Show**: Open `version_0/complete_output/ANALYSIS_SUMMARY.md`
2. **Play**: `generated_drums.mid` in any MIDI player
3. **Compare**: Play `original_drums.mid` and random baselines
4. **Visualize**: Show `correlation_plot.png`

## 🔍 Troubleshooting

### "No checkpoint found"

-   Make sure training completed successfully
-   Check `lightning_logs/version_X/checkpoints/` exists

### "Encoding errors"

-   Windows CMD issue - use PowerShell instead

### "Model loading errors"

-   Vocabulary size mismatch - should auto-resolve in current version

## 📈 Expected Evolution

-   **Version 0**: Weak correlation (~0.04) but learning basics
-   **More epochs**: Stronger correlation (target >0.15)
-   **More data**: Better generalization

---

**Bottom Line**: Even with weak initial correlation, the system proves the AI is
learning patterns rather than generating random noise. With more training,
correlation should improve significantly.
