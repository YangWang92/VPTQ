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

def filter_and_categorize_layers(layer_dist, high_prob_threshold=0.01, low_prob_threshold=0.001):
    """
    Filter and categorize layer values based on probability thresholds.
    - Values >= high_prob_threshold: Use 3 bits
    - Values between low_prob_threshold and high_prob_threshold: Use 2 bits
    - Values < low_prob_threshold: Filter out (don't include)
    
    Returns:
    - filtered_dist: Dictionary with filtered values
    - stats: Statistics about the filtering process
    """
    filtered_dist = {}
    stats = {
        'total_values': 0,
        'high_prob_values': 0,  # 3 bits
        'low_prob_values': 0,   # 2 bits
        'filtered_values': 0,   # removed
        'memory_savings': 0,    # estimated memory savings
    }
    
    for layer, values in layer_dist.items():
        # Normalize values to ensure they sum to 1 (100%)
        total = sum(values)
        normalized_values = [v/total for v in values]
        
        # Initialize lists for different categories
        high_prob = []
        low_prob = []
        
        # Categorize values
        for i, prob in enumerate(normalized_values):
            stats['total_values'] += 1
            
            if prob >= high_prob_threshold:
                high_prob.append((i, values[i]))
                stats['high_prob_values'] += 1
            elif prob >= low_prob_threshold:
                low_prob.append((i, values[i]))
                stats['low_prob_values'] += 1
            else:
                stats['filtered_values'] += 1
        
        # Only include layers with sufficient high probability values
        if high_prob or low_prob:
            filtered_dist[layer] = {
                'high_prob': high_prob,  # 3 bits
                'low_prob': low_prob     # 2 bits
            }
    
    # Calculate memory savings (original: 8 bits per value)
    original_bits = stats['total_values'] * 8
    new_bits = (stats['high_prob_values'] * 3) + (stats['low_prob_values'] * 2)
    stats['memory_savings'] = 1 - (new_bits / original_bits)
    
    return filtered_dist, stats

def main():
    if len(sys.argv) < 2:
        print("Usage: python token_dist.py <file_path> [high_prob_threshold] [low_prob_threshold]")
        return
    
    file_path = sys.argv[1]
    
    # Optional command-line arguments for thresholds
    high_prob_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    low_prob_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
    
    layer_dist = read_layer_dist(file_path)
    
    # Apply filtering and categorization
    filtered_dist, stats = filter_and_categorize_layers(
        layer_dist, 
        high_prob_threshold=high_prob_threshold,
        low_prob_threshold=low_prob_threshold
    )
    
    # Print statistics
    print("\nFiltering Statistics:")
    print(f"Total values: {stats['total_values']}")
    print(f"High probability values (3 bits): {stats['high_prob_values']} ({stats['high_prob_values']/stats['total_values']*100:.2f}%)")
    print(f"Low probability values (2 bits): {stats['low_prob_values']} ({stats['low_prob_values']/stats['total_values']*100:.2f}%)")
    print(f"Filtered out values: {stats['filtered_values']} ({stats['filtered_values']/stats['total_values']*100:.2f}%)")
    print(f"Estimated memory savings: {stats['memory_savings']*100:.2f}%")
    
    # Print filtered distribution summary
    print("\nFiltered Layer Distribution:")
    for layer, categories in filtered_dist.items():
        high_count = len(categories['high_prob'])
        low_count = len(categories['low_prob'])
        total = high_count + low_count
        print(f"{layer}: {total} values total ({high_count} high-prob, {low_count} low-prob)")
    
    # Save the filtered distribution to a new file
    output_file = file_path.replace('.jsonl', '_filtered.json')
    with open(output_file, 'w') as f:
        json.dump(filtered_dist, f, indent=2)
    
    print(f"\nFiltered distribution saved to {output_file}")

if __name__ == "__main__":
    main()