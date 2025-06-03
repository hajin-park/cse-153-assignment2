#!/usr/bin/env python
"""
Debug the exact data loading issue
"""
import torch
from torch.utils.data import DataLoader
from choreo2groove import ChoreoGrooveDataset, collate_fn, VOCAB_SIZE, DRUM_TOKENS

def debug_single_sample():
    """Debug a single sample from the dataset"""
    print("🔍 Debugging single sample...")
    
    ds = ChoreoGrooveDataset("dataset_root", seq_len=256)
    print(f"Dataset size: {len(ds)}")
    
    # Check first sample
    pose, tokens = ds[0]
    print(f"Pose shape: {pose.shape}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Tokens dtype: {tokens.dtype}")
    print(f"Token min: {tokens.min().item()}")
    print(f"Token max: {tokens.max().item()}")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")
    
    # Check for out-of-range tokens
    out_of_range = tokens >= VOCAB_SIZE
    if out_of_range.any():
        print(f"❌ Found {out_of_range.sum()} out-of-range tokens!")
        bad_tokens = tokens[out_of_range]
        print(f"Bad token values: {bad_tokens[:10]}")  # Show first 10
    else:
        print("✅ All tokens are in valid range")
    
    return pose, tokens

def debug_dataloader():
    """Debug the dataloader with collate function"""
    print("\\n🔍 Debugging dataloader...")
    
    ds = ChoreoGrooveDataset("dataset_root", seq_len=256)
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    try:
        batch = next(iter(dl))
        poses, tokens = batch
        print(f"Batch poses shape: {poses.shape}")
        print(f"Batch tokens shape: {tokens.shape}")
        print(f"Batch tokens min: {tokens.min().item()}")
        print(f"Batch tokens max: {tokens.max().item()}")
        
        # Check for out-of-range tokens in batch
        out_of_range = tokens >= VOCAB_SIZE
        if out_of_range.any():
            print(f"❌ Found {out_of_range.sum()} out-of-range tokens in batch!")
            return False
        else:
            print("✅ All batch tokens are in valid range")
            return True
            
    except Exception as e:
        print(f"💥 Error in dataloader: {e}")
        return False

def debug_model_forward():
    """Debug model forward pass with actual data"""
    print("\\n🔍 Debugging model forward pass...")
    
    from choreo2groove import Choreo2GrooveModel
    
    # Get a batch
    ds = ChoreoGrooveDataset("dataset_root", seq_len=256)
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    poses, tokens = next(iter(dl))
    
    print(f"Input poses shape: {poses.shape}")
    print(f"Input tokens shape: {tokens.shape}")
    
    # Create model
    in_feats = poses.shape[-1]
    model = Choreo2GrooveModel(in_feats)
    model.eval()
    
    try:
        with torch.no_grad():
            # Try the encoder
            print("Testing encoder...")
            memory = model.encoder(poses)
            print(f"Encoder output shape: {memory.shape}")
            
            # Try the decoder with reduced tokens
            print("Testing decoder...")
            decoder_input = tokens[:, :-1]  # Remove last token for decoder input
            print(f"Decoder input shape: {decoder_input.shape}")
            print(f"Decoder input range: [{decoder_input.min()}, {decoder_input.max()}]")
            
            # Check if decoder input is in range
            if decoder_input.max() >= VOCAB_SIZE:
                print(f"❌ Decoder input has out-of-range tokens! Max: {decoder_input.max()}, VOCAB_SIZE: {VOCAB_SIZE}")
                return False
            
            logits = model.decoder(decoder_input, memory)
            print(f"Decoder output shape: {logits.shape}")
            print("✅ Forward pass successful!")
            return True
            
    except Exception as e:
        print(f"💥 Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🐛 DEBUGGING CHOREO2GROOVE DATA PIPELINE\\n")
    
    # Step 1: Check single sample
    pose, tokens = debug_single_sample()
    
    # Step 2: Check dataloader
    dl_ok = debug_dataloader()
    
    # Step 3: Check model forward pass
    if dl_ok:
        model_ok = debug_model_forward()
        if model_ok:
            print("\\n🎉 ALL TESTS PASSED! Training should work now.")
        else:
            print("\\n❌ Model forward pass failed.")
    else:
        print("\\n❌ Dataloader failed.")
    
    print(f"\\nFinal check - VOCAB_SIZE: {VOCAB_SIZE}")
    print(f"Pad token ID: {DRUM_TOKENS['pad']}")
    print(f"Max shift token: {DRUM_TOKENS.get('shift_100', 'not found')}") 