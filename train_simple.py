#!/usr/bin/env python
"""
Simplified training script for Choreo2Groove without PyTorch Lightning
This helps debug the exact issue with the model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from choreo2groove import ChoreoGrooveDataset, collate_fn, VOCAB_SIZE, PoseEncoder, DrumDecoder

class SimpleModel(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.encoder = PoseEncoder(in_feats)
        self.decoder = DrumDecoder()
        
    def forward(self, poses, drums):
        memory = self.encoder(poses)  # (T, B, E)
        logits = self.decoder(drums, memory)  # Full sequence
        return logits

def debug_batch(batch_idx, poses, tokens):
    """Debug a specific batch to see what's wrong"""
    print(f"\\n🔍 DEBUGGING BATCH {batch_idx}")
    print(f"Poses shape: {poses.shape}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Tokens min: {tokens.min()}, max: {tokens.max()}")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")
    
    # Check for problematic tokens
    bad_mask = tokens >= VOCAB_SIZE
    if bad_mask.any():
        print(f"❌ Found {bad_mask.sum()} bad tokens >= {VOCAB_SIZE}")
        bad_indices = torch.where(bad_mask)
        print(f"Bad token positions: {list(zip(bad_indices[0][:5].tolist(), bad_indices[1][:5].tolist()))}")
        print(f"Bad token values: {tokens[bad_mask][:10].tolist()}")
        return False
    else:
        print("✅ All tokens are valid")
        return True

def train_simple():
    """Simple training loop without PyTorch Lightning"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset_root")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    print(f"🚀 Starting simple training with {args.epochs} epochs...")
    
    # Setup data
    ds = ChoreoGrooveDataset(args.data_root, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, 
                   collate_fn=collate_fn, num_workers=0)
    
    print(f"Dataset size: {len(ds)}")
    print(f"Batches per epoch: {len(dl)}")
    
    # Setup model
    sample_pose, _ = ds[0]
    in_feats = sample_pose.shape[-1]
    print(f"Input features: {in_feats}")
    
    model = SimpleModel(in_feats)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    
    print(f"Model created with VOCAB_SIZE: {VOCAB_SIZE}")
    
    # Training loop
    model.train()
    
    try:
        for epoch in range(args.epochs):
            print(f"\\n📊 Epoch {epoch+1}/{args.epochs}")
            epoch_loss = 0
            
            for batch_idx, (poses, tokens) in enumerate(tqdm(dl, desc=f"Epoch {epoch+1}")):
                
                # Debug the batch
                is_valid = debug_batch(batch_idx, poses, tokens)
                if not is_valid:
                    print(f"💥 Skipping invalid batch {batch_idx}")
                    continue
                
                try:
                    # Forward pass with debugging
                    print(f"Forward pass - input tokens shape: {tokens[:, :-1].shape}")
                    print(f"Target tokens shape: {tokens[:, 1:].shape}")
                    
                    # Check input tokens are in range
                    input_tokens = tokens[:, :-1]
                    target_tokens = tokens[:, 1:]
                    
                    if input_tokens.max() >= VOCAB_SIZE:
                        print(f"❌ Input tokens out of range: max={input_tokens.max()}")
                        continue
                        
                    logits = model(poses, input_tokens)
                    print(f"Logits shape: {logits.shape}")
                    
                    # Calculate loss
                    loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_tokens.reshape(-1))
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    print(f"✅ Batch {batch_idx} completed, loss: {loss.item():.4f}")
                    
                except Exception as e:
                    print(f"💥 Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            avg_loss = epoch_loss / len(dl)
            print(f"📈 Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    except KeyboardInterrupt:
        print("\\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n🏁 Training completed!")

if __name__ == "__main__":
    train_simple() 