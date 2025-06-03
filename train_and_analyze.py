#!/usr/bin/env python
"""
Integrated Training and Analysis Script for Choreo2Groove
- Train with configurable parameters (epochs, learning rate, etc.)
- Automatically generate complete analysis package after training
- Smart version detection and management
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from choreo2groove import Choreo2GrooveModel, ChoreoGrooveDataset

def find_next_version():
    """Find the next available version number"""
    lightning_logs = Path("lightning_logs")
    if not lightning_logs.exists():
        return 0
    
    versions = []
    for version_dir in lightning_logs.glob("version_*"):
        try:
            version_num = int(version_dir.name.split("_")[1])
            versions.append(version_num)
        except:
            continue
    
    return max(versions) + 1 if versions else 0

def train_model(epochs=5, lr=1e-4, batch_size=4, seq_len=256, version=None):
    """Train the model with specified parameters"""
    
    if version is None:
        version = find_next_version()
    
    print("🚀 CHOREO2GROOVE TRAINING WITH AUTO-ANALYSIS")
    print("=" * 60)
    print(f"🔢 Version: {version}")
    print(f"📊 Epochs: {epochs}")
    print(f"🎯 Learning Rate: {lr}")
    print(f"📦 Batch Size: {batch_size}")
    print(f"📏 Sequence Length: {seq_len}")
    print("=" * 60)
    
    # Setup data
    print("🔄 Loading dataset...")
    dataset = ChoreoGrooveDataset("dataset_root", seq_len=seq_len)
    
    if len(dataset) == 0:
        print("❌ No dataset found! Run data processing first.")
        return None
    
    # Calculate input features from first sample
    sample_pose, _ = dataset[0]
    in_feats = sample_pose.shape[-1]
    print(f"✅ Dataset loaded: {len(dataset)} samples, {in_feats} features per frame")
    
    # Setup model
    print("🧠 Creating model...")
    model = Choreo2GrooveModel(
        in_feats=in_feats,
        lr=lr,
        batch_size=batch_size,
        seq_len=seq_len
    )
    
    # Setup callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        filename='choreo2groove-{epoch:02d}-{train_loss:.3f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    
    logger = TensorBoardLogger("lightning_logs", version=version)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=10,
        check_val_every_n_epoch=1
    )
    
    # Start training
    print(f"🎵 Starting training for {epochs} epochs...")
    start_time = datetime.now()
    
    try:
        trainer.fit(model, dataset)
        training_success = True
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        print(f"\n✅ Training completed successfully!")
        print(f"⏱️  Training duration: {training_duration}")
        
        # Get final metrics
        final_loss = trainer.callback_metrics.get('train_loss', 'unknown')
        print(f"📈 Final training loss: {final_loss}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        training_success = False
        return None
    
    return version, training_success

def run_complete_analysis(version, sample_idx=58):
    """Run the complete analysis package generation"""
    print("\n" + "="*60)
    print("🔍 STARTING AUTOMATIC ANALYSIS GENERATION")
    print("="*60)
    
    try:
        # Import and run the analysis generation
        from generate_everything import generate_everything_for_version
        
        print(f"🎯 Generating complete analysis for version {version}...")
        result = generate_everything_for_version(version=version, sample_idx=sample_idx)
        
        if result:
            print(f"\n🎉 COMPLETE SUCCESS!")
            print(f"📁 Analysis package location: {result}")
            return result
        else:
            print(f"\n❌ Analysis generation failed")
            return None
            
    except Exception as e:
        print(f"\n❌ Analysis generation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Train Choreo2Groove and generate complete analysis")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, 
                       help="Number of training epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size (default: 4)")
    parser.add_argument("--seq_len", type=int, default=256, 
                       help="Sequence length (default: 256)")
    
    # Analysis parameters
    parser.add_argument("--sample_idx", type=int, default=58, 
                       help="Sample index for analysis (default: 58)")
    parser.add_argument("--version", type=int, default=None, 
                       help="Force specific version number (auto-detect if not specified)")
    
    # Control flags
    parser.add_argument("--train_only", action="store_true", 
                       help="Only train, skip analysis generation")
    parser.add_argument("--analyze_only", action="store_true", 
                       help="Only run analysis on existing version")
    
    args = parser.parse_args()
    
    print("🎵 CHOREO2GROOVE INTEGRATED TRAINING & ANALYSIS")
    print("=" * 70)
    
    if args.analyze_only:
        # Only run analysis
        if args.version is None:
            print("❌ Must specify --version when using --analyze_only")
            return
        
        print(f"🔍 Running analysis only for version {args.version}")
        result = run_complete_analysis(args.version, args.sample_idx)
        
        if result:
            print(f"\n✨ Analysis complete! Check: {result}")
        else:
            print(f"\n❌ Analysis failed!")
        
        return
    
    # Run training
    print("🚀 Starting training phase...")
    training_result = train_model(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        version=args.version
    )
    
    if training_result is None:
        print("❌ Training failed, stopping here.")
        return
    
    version, training_success = training_result
    
    if not training_success:
        print("❌ Training unsuccessful, skipping analysis.")
        return
    
    if args.train_only:
        print(f"\n✅ Training complete for version {version}. Skipping analysis as requested.")
        print(f"💡 To run analysis later: python train_and_analyze.py --analyze_only --version {version}")
        return
    
    # Run automatic analysis
    print(f"\n🔄 Training complete! Starting automatic analysis generation...")
    analysis_result = run_complete_analysis(version, args.sample_idx)
    
    if analysis_result:
        print(f"\n🎊 COMPLETE SUCCESS!")
        print(f"🎯 Training version: {version}")
        print(f"📁 Complete package: {analysis_result}")
        print(f"\n📖 Next steps:")
        print(f"   1. Open: {analysis_result}/DEMO_SCRIPT.md")
        print(f"   2. Test: Play generated_drums.mid with dance_video_FIXED.gif")
        print(f"   3. Analyze: Check alignment_analysis_sample_{args.sample_idx}.png")
    else:
        print(f"\n⚠️  Training succeeded but analysis failed.")
        print(f"💡 You can run analysis manually:")
        print(f"   python train_and_analyze.py --analyze_only --version {version}")

def quick_commands():
    """Print quick command examples"""
    print("🎯 QUICK COMMAND EXAMPLES:")
    print("=" * 50)
    print("# Default training (5 epochs, lr=1e-4)")
    print("python train_and_analyze.py")
    print()
    print("# Custom parameters")
    print("python train_and_analyze.py --epochs 10 --lr 2e-4 --batch_size 8")
    print()
    print("# Train only, skip analysis")
    print("python train_and_analyze.py --epochs 3 --train_only")
    print()
    print("# Analyze existing version")
    print("python train_and_analyze.py --analyze_only --version 6")
    print()
    print("# Use different sample for analysis")
    print("python train_and_analyze.py --sample_idx 25")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🎵 Choreo2Groove Integrated Training & Analysis")
        print("No arguments provided. Here are some examples:")
        print()
        quick_commands()
        print()
        print("Add --help for full argument list")
    else:
        main() 