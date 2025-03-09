import json
import sys

def calculate_layer_sums(file_path):
    # Dictionary to store the sum for each layer
    layer_sums = {}
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Parse the JSON object in each line
                data = json.loads(line.strip())
                
                # Extract the layer and its values
                for layer_name, values in data.items():
                    if layer_name not in layer_sums:
                        layer_sums[layer_name] = 0
                    
                    # Add the sum of values for this layer
                    layer_sums[layer_name] += sum(values)
            except json.JSONDecodeError:
                print(f"Error parsing JSON in line: {line[:100]}...")
                continue
    
    return layer_sums

def main():
    if len(sys.argv) < 2:
        print("Usage: python dist.py <file_path>")
        return
    
    file_path = sys.argv[1]
    layer_sums = calculate_layer_sums(file_path)
    
    # Sort layers by their number for better readability
    sorted_layers = sorted(layer_sums.keys(), 
                          key=lambda x: int(x.split()[1]) if len(x.split()) > 1 and x.split()[1].isdigit() else 0)
    
    # Print the results
    for layer in sorted_layers:
        print(f"{layer}: {layer_sums[layer]}")

if __name__ == "__main__":
    main()