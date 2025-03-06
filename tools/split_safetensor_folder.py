import safetensors
import argparse
import os
import json
import math
from safetensors.torch import save_file
import glob

# load large safetensor files from a directory
# split each file into smaller ones
# save them to disk to meet huggingface safetensors file format
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing safetensor shards")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save split safetensors")
    parser.add_argument("--max-shard-size", type=int, default=5 * 1024 * 1024 * 1024, help="Maximum shard size in bytes (default: 1GB)")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of shards to split each input file into (overrides max-shard-size if provided)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all safetensor files in the input directory
    input_files = glob.glob(os.path.join(args.input_dir, "*.safetensors"))
    
    if not input_files:
        print(f"No safetensor files found in {args.input_dir}")
        exit(1)
    
    print(f"Found {len(input_files)} safetensor files to process")
    
    # Process each input file
    for input_file in input_files:
        # Create a subdirectory for each input file's outputs
        input_file_basename = os.path.basename(input_file)
        input_file_name = os.path.splitext(input_file_basename)[0]
        output_subdir = os.path.join(args.output_dir, input_file_name)
        os.makedirs(output_subdir, exist_ok=True)
        
        print(f"\nProcessing {input_file}...")
        print(f"Output will be saved to {output_subdir}")
        
        # Load the safetensor file
        try:
            model_data = safetensors.torch.load_file(input_file)
        except Exception as e:
            print(f"Error loading {input_file}: {e}")
            continue

        # Calculate total size of the model
        total_size = 0
        for tensor in model_data.values():
            total_size += tensor.numel() * tensor.element_size()

        print(f"Total size of {input_file_basename}: {total_size / (1024 * 1024 * 1024):.2f} GB")

        # Determine number of shards
        if args.num_shards is not None:
            num_shards = args.num_shards
        else:
            num_shards = math.ceil(total_size / args.max_shard_size)

        print(f"Splitting {input_file_basename} into {num_shards} shards...")

        # Create weight map for index.json
        weight_map = {}
        metadata = {"total_size": total_size}

        # Group tensors into shards
        tensors_per_shard = math.ceil(len(model_data) / num_shards)

        # Split tensors into shards
        for shard_idx in range(num_shards):
            shard_name = f"{input_file_name}-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"
            shard_path = os.path.join(output_subdir, shard_name)

            # Get tensors for this shard
            start_idx = shard_idx * tensors_per_shard
            end_idx = min((shard_idx + 1) * tensors_per_shard, len(model_data))

            shard_tensors = {}
            keys = list(model_data.keys())

            for i in range(start_idx, end_idx):
                key = keys[i]
                shard_tensors[key] = model_data[key]
                weight_map[key] = shard_name

            print(f"Saving shard {shard_idx+1}/{num_shards} with {len(shard_tensors)} tensors to {shard_path}")
            save_file(shard_tensors, shard_path)

        # Create and save index.json
        index = {
            "metadata": metadata,
            "weight_map": weight_map
        }

        index_path = os.path.join(output_subdir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"Successfully split {input_file_basename} into {num_shards} shards and created index file at {index_path}")
    
    print("\nAll files processed successfully!")