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
    
    # Find all safetensor files in the input directory
    safetensor_files = glob.glob(os.path.join(args.input_dir, "*.safetensors"))
    if not safetensor_files:
        # Check if there's an index file to guide the merging
        index_file = os.path.join(args.input_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            print(f"Found index file: {index_file}")
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Get all unique shard files from the weight map
            shard_files = set(index_data["weight_map"].values())
            safetensor_files = [os.path.join(args.input_dir, shard) for shard in shard_files]
        else:
            # Try to find files with pattern model-*-of-*.safetensors
            safetensor_files = glob.glob(os.path.join(args.input_dir, "model-*-of-*.safetensors"))
            
    if not safetensor_files:
        # Try to find files with pattern model*-mp*.safetensors
        safetensor_files = glob.glob(os.path.join(args.input_dir, "model*-mp*.safetensors"))
    
    if not safetensor_files:
        raise ValueError(f"No safetensor files found in {args.input_dir}")
    
    # Sort files naturally to ensure correct order
    safetensor_files.sort(key=natural_sort_key)
    
    print(f"Found {len(safetensor_files)} safetensor files to merge")
    for file in safetensor_files:
        print(f"  - {os.path.basename(file)}")
    
    # Load all tensors from all files
    all_tensors = {}
    for file in safetensor_files:
        print(f"Loading tensors from {os.path.basename(file)}...")
        tensors = load_file(file)
        all_tensors.update(tensors)
    
    print(f"Loaded a total of {len(all_tensors)} tensors")
    
    # Calculate total size
    total_size = 0
    for tensor in all_tensors.values():
        total_size += tensor.numel() * tensor.element_size()
    
    print(f"Total model size: {total_size / (1024 * 1024 * 1024):.2f} GB")
    
    # Determine how to distribute tensors across output shards
    num_shards = args.num_output_shards
    tensors_per_shard = len(all_tensors) // num_shards
    remainder = len(all_tensors) % num_shards
    
    print(f"Distributing tensors across {num_shards} output shards")
    
    # Create weight map for index.json
    weight_map = {}
    metadata = {"total_size": total_size}
    
    # Distribute tensors across shards
    keys = list(all_tensors.keys())
    start_idx = 0
    
    for shard_idx in range(num_shards):
        # Calculate how many tensors go in this shard
        shard_tensor_count = tensors_per_shard + (1 if shard_idx < remainder else 0)
        end_idx = start_idx + shard_tensor_count
        
        shard_name = f"model{shard_idx}-mp{num_shards}.safetensors"
        shard_path = os.path.join(args.output_dir, shard_name)
        
        shard_tensors = {}
        for i in range(start_idx, end_idx):
            if i < len(keys):
                key = keys[i]
                shard_tensors[key] = all_tensors[key]
                weight_map[key] = shard_name
        
        print(f"Saving shard {shard_idx+1}/{num_shards} with {len(shard_tensors)} tensors to {shard_path}")
        save_file(shard_tensors, shard_path)
        
        start_idx = end_idx
    
    # Create and save index.json
    index = {
        "metadata": metadata,
        "weight_map": weight_map
    }
    
    index_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"Successfully merged tensors into {num_shards} shards and created index file at {index_path}")
