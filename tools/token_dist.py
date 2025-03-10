import json
import sys
import numpy as np

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python token_dist.py <file_path> [threshold]")
        return
    
    file_path = sys.argv[1]
    
    # Optional command-line argument for threshold
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    
    layer_dist = read_layer_dist(file_path)
    
    # Apply filtering with single threshold
    filtered_dist, stats = filter_layers(
        layer_dist, 
        threshold=threshold
    )
    
    # Print statistics
    print("\nFiltering Statistics:")
    print(f"Total values: {stats['total_values']}")
    print(f"Kept values: {stats['kept_values']} ({stats['kept_values']/stats['total_values']*100:.2f}%)")
    print(f"Filtered out values: {stats['filtered_values']} ({stats['filtered_values']/stats['total_values']*100:.2f}%)")
    print(f"Estimated memory savings: {stats['memory_savings']*100:.2f}%")
    
    # Print filtered distribution summary
    print("\nFiltered Layer Distribution:")
    for layer, expert_ids in filtered_dist.items():
        print(f"Layer {layer}: {len(expert_ids)} experts with high distribution")
        # print(f"Layer {layer}: {len(expert_ids)} experts with high distribution, {expert_ids}")
 
    # Save the filtered distribution to a new file
    output_file = file_path.replace('.jsonl', '_filtered.json')
    with open(output_file, 'w') as f:
        json.dump(filtered_dist, f, indent=2)
    
    print(f"\nFiltered distribution saved to {output_file}")

if __name__ == "__main__":
    main()