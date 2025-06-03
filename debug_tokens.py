#!/usr/bin/env python
"""
Debug script to check drum tokenization
"""
import os
import numpy as np
import pretty_midi as pm
from glob import glob

# Import the tokenization from main script
from choreo2groove import DRUM_TOKENS, VOCAB_SIZE, midi_to_tokens

def debug_dataset():
    """Check all drum files for tokenization issues"""
    drum_files = sorted(glob("dataset_root/*/drums.mid"))
    print(f"Checking {len(drum_files)} drum files...")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Valid token range: 0 to {VOCAB_SIZE-1}")
    
    max_token = 0
    problematic_files = []
    
    for i, drum_file in enumerate(drum_files[:10]):  # Check first 10
        try:
            midi = pm.PrettyMIDI(drum_file)
            tokens = midi_to_tokens(midi)
            
            if tokens:
                file_max = max(tokens)
                max_token = max(max_token, file_max)
                
                if file_max >= VOCAB_SIZE:
                    problematic_files.append((drum_file, file_max, tokens[:10]))
                    print(f"❌ {drum_file}: max token {file_max} >= vocab size {VOCAB_SIZE}")
                else:
                    print(f"✅ {drum_file}: max token {file_max} < vocab size {VOCAB_SIZE}")
            else:
                print(f"⚠️  {drum_file}: empty tokens")
                
        except Exception as e:
            print(f"💥 {drum_file}: error {e}")
    
    print(f"\\nOverall max token found: {max_token}")
    print(f"Problematic files: {len(problematic_files)}")
    
    if problematic_files:
        print("\\nFirst problematic file details:")
        file, max_tok, sample_tokens = problematic_files[0]
        print(f"File: {file}")
        print(f"Max token: {max_tok}")
        print(f"Sample tokens: {sample_tokens}")
    
    return max_token

def fix_vocab_size():
    """Calculate the correct vocab size needed"""
    max_token = debug_dataset()
    needed_vocab = max_token + 1
    current_vocab = VOCAB_SIZE
    
    print(f"\\n🔧 SOLUTION:")
    print(f"Current VOCAB_SIZE: {current_vocab}")
    print(f"Needed VOCAB_SIZE: {needed_vocab}")
    print(f"Difference: {needed_vocab - current_vocab}")
    
    if needed_vocab > current_vocab:
        print(f"\\n⚠️  Need to increase VOCAB_SIZE to {needed_vocab}")
        return needed_vocab
    else:
        print(f"\\n✅ Current VOCAB_SIZE is sufficient")
        return current_vocab

if __name__ == "__main__":
    needed_size = fix_vocab_size()
    print(f"\\nRecommended VOCAB_SIZE = {needed_size}") 