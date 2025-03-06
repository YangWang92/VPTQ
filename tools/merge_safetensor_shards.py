import safetensors
import argparse
import os
import json
import glob
from safetensors.torch import load_file, save_file
import re

def natural_sort_key(s):
    """Sort strings containing numbers naturally"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge split safetensor files back into original shards")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing split safetensor subdirectories")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save merged safetensor shards")
    parser.add_argument("--num-output-shards", type=int, default=4, help="Number of output shards to create (default: 4)")
    args = parser.parse_args()

    # Create output directory if it doesn"t exist
    os.makedirs(args.output_dir, exist_ok=True)
