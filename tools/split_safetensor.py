import safetensors
import argparse
import os
import json
import math
from safetensors.torch import save_file

# load large safetensor file
# split it into smaller ones
# save them to disk to meet huggingface safetensors file format
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input-file", type=str, required=True)
	parser.add_argument("--output-dir", type=str, required=True)
	parser.add_argument("--max-shard-size", type=int, default=6 * 1024 * 1024 * 1024, help="Maximum shard size in bytes (default: 10GB)")
	parser.add_argument("--num-shards", type=int, default=None, help="Number of shards to split into (overrides max-shard-size if provided)")
	args = parser.parse_args()
	
	# Create output directory if it doesn't exist
	os.makedirs(args.output_dir, exist_ok=True)
	
	print(f"Loading model from {args.input_file}...")
	model_data = safetensors.torch.load_file(args.input_file)
	
	# Calculate total size of the model
	total_size = 0
	for tensor in model_data.values():
		total_size += tensor.numel() * tensor.element_size()
	
	print(f"Total model size: {total_size / (1024 * 1024 * 1024):.2f} GB")
	
	# Determine number of shards
	if args.num_shards is not None:
		num_shards = args.num_shards
	else:
		num_shards = math.ceil(total_size / args.max_shard_size)
	
	print(f"Splitting model into {num_shards} shards...")
	
	# Create weight map for index.json
	weight_map = {}
	metadata = {"total_size": total_size}
	
	# Group tensors into shards
	tensors_per_shard = math.ceil(len(model_data) / num_shards)
	
	# Split tensors into shards
	for shard_idx in range(num_shards):
		shard_name = f"model-{shard_idx+1:05d}-of-{num_shards:06d}.safetensors"
		shard_path = os.path.join(args.output_dir, shard_name)
		
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
	
	index_path = os.path.join(args.output_dir, "model.safetensors.index.json")
	with open(index_path, "w") as f:
		json.dump(index, f, indent=2)
	
	print(f"Successfully split model into {num_shards} shards and created index file at {index_path}")
	
	
