import safetensors.torch
import torch
import json
import os

def read_layer_dist(file_path):
    layer_dist = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            # Update the layer_dist dictionary with the data from each line
            for layer_key, values in data.items():
                layer_key = int(layer_key.split(' ')[1])
                if layer_key in layer_dist:
                    # If the layer already exists in our dictionary, extend its values
                    layer_dist[layer_key].extend(values)
                else:
                    # Otherwise, create a new entry
                    layer_dist[layer_key] = values
    return layer_dist

def filter_layers(layer_dist, threshold=0.01):
    """
    Filter layer values based on a single probability threshold.
    - Values >= threshold: Keep
    - Values < threshold: Filter out
    
    Returns:
    - filtered_dist: Dictionary with {layer_idx: [high_dist_expert_ids]}
    - stats: Statistics about the filtering process
    """
    filtered_dist = {}
    stats = {
        'total_values': 0,
        'kept_values': 0,     # values above threshold
        'filtered_values': 0,  # values below threshold
        'memory_savings': 0,   # estimated memory savings
    }
    
    for layer, values in layer_dist.items():
        # Normalize values to ensure they sum to 1 (100%)
        total = sum(values)
        normalized_values = [v/total for v in values]
        
        # Initialize list for high distribution expert IDs
        high_dist_expert_ids = []
        
        # Filter values based on threshold
        for i, prob in enumerate(normalized_values):
            stats['total_values'] += 1
            
            if prob >= threshold:
                high_dist_expert_ids.append(i)  # Only store the expert ID
                stats['kept_values'] += 1
            else:
                stats['filtered_values'] += 1
        
        # Only include layers with high distribution experts
        if high_dist_expert_ids:
            filtered_dist[layer] = high_dist_expert_ids
    
    # Calculate memory savings (original: 8 bits per value)
    original_bits = stats['total_values'] * 8
    # Assuming we use 4 bits for kept values (simplified from previous approach)
    new_bits = stats['kept_values'] * 4
    stats['memory_savings'] = 1 - (new_bits / original_bits)
    
    return filtered_dist, stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_0_path", type=str, required=True)
    parser.add_argument("--model_1_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--layer_dist_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=False, default=0.002)
    args = parser.parse_args()
    
    layer_dist = read_layer_dist(args.layer_dist_path)
    filtered_dist, stats = filter_layers(
        layer_dist, 
        threshold=args.threshold
    )
    for layer, expert_ids in filtered_dist.items():
        print(f'Layer {layer}: {len(expert_ids)} experts with high distribution')
    
    os.makedirs(args.output_path, exist_ok=True)
    # number of shards
    # policy: merge w2 from model_1 to model_0
    for i in range(args.num_shards):
        print(f'Merging model {i}...')
        model_0 = safetensors.torch.load_file(f'{args.model_0_path}/model{i}-mp4.safetensors')
        model_1 = safetensors.torch.load_file(f'{args.model_1_path}/model{i}-mp4.safetensors')
        for key in model_1.keys():
            # layer_idx = int(key.split('.')[0].split('_')[-1])
            len_key = len(key.split('.'))
            if '.experts.' in key:
                layer_idx = int(key.split('.')[1])
                expert_idx = int(key.split('.')[4])
                if expert_idx in filtered_dist[layer_idx]:
                    model_0[key] = model_1[key]
            elif 'w2' in key or 'wq' in key or 'wk' in key or 'wo' in key or 'shared' in key:
                model_0[key] = model_1[key]
        safetensors.torch.save_file(model_0, f'{args.output_path}/model{i}-mp4.safetensors')
    
        model_0_config = json.load(open(f'{args.model_0_path}/config.json'))
        model_1_config = json.load(open(f'{args.model_1_path}/config.json'))
        for key in model_1_config['quantization_config']['config_for_layers'].keys():
            if '.experts.' in key:
                layer_idx = int(key.split('.')[1])
                expert_idx = int(key.split('.')[4])
                if expert_idx in filtered_dist[layer_idx]:
                    model_0_config['quantization_config']['config_for_layers'][key] = model_1_config['quantization_config']['config_for_layers'][key]
            elif 'w2' in key or 'wq' in key or 'wk' in key or 'wo' in key or 'shared' in key:
                model_0_config['quantization_config']['config_for_layers'][key] = model_1_config['quantization_config']['config_for_layers'][key]
        with open(f'{args.output_path}/config.json', 'w') as f:
            json.dump(model_0_config, f)
