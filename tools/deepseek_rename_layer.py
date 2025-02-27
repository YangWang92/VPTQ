import safetensors.torch
import os
import argparse

# load ~/yangwang/packed_output_model/model0-mp1.safetensors

mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}

reverse_mapping = {
    "embed": "embed_tokens",
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "wq": "q_proj",
    "wq_a": "q_a_proj",
    "q_norm": "q_a_layernorm",
    "wq_b": "q_b_proj",
    "wkv_a": "kv_a_proj_with_mqa",
    "kv_norm": "kv_a_layernorm",
    "wkv_b": "kv_b_proj",
    "wo": "o_proj",
    "gate": "gate",
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
    "norm": "norm",
    "head": "lm_head",
    "scale": "scale",
}

# Additional mappings for component names
component_mapping = {
    "attn": "self_attn",
    "ffn": "mlp"
}

# Reverse component mapping
reverse_component_mapping = {v: k for k, v in component_mapping.items()}

# Function to convert shortened layer names back to original format
def convert_to_original_format(shortened_key):
    parts = shortened_key.split('.')
    
    # Handle special cases first
    if shortened_key == "embed.weight":
        return "model.embed_tokens.weight"
    elif shortened_key == "head.weight":
        return "lm_head.weight"
    elif shortened_key == "norm.weight" or shortened_key.startswith("norm."):
        # Handle norm.weight and other norm.* keys
        return f"model.{shortened_key}"
    
    # Handle layer-specific transformations
    if parts[0] == "layers":
        layer_num = parts[1]
        
        # Special handling for layernorm components
        if len(parts) >= 3 and parts[2] in ["attn_norm", "ffn_norm"]:
            if parts[2] == "attn_norm":
                return f"model.layers.{layer_num}.input_layernorm.{parts[3]}"
            elif parts[2] == "ffn_norm":
                return f"model.layers.{layer_num}.post_attention_layernorm.{parts[3]}"
        
        component = parts[2]
        
        # Start building the original key
        original_key = ["model", "layers", layer_num]
        
        # Convert component name (e.g., ffn -> mlp)
        if component in component_mapping:
            original_key.append(component_mapping[component])
        else:
            original_key.append(component)
        
        # Process the rest of the path
        i = 3
        while i < len(parts):
            if parts[i] == "experts" and i + 1 < len(parts):
                # Handle experts section
                original_key.append("experts")
                original_key.append(parts[i+1])
                i += 2
            elif parts[i] in reverse_mapping:
                # Convert layer name (e.g., w3 -> up_proj)
                original_key.append(reverse_mapping[parts[i]])
                
                # Add appropriate suffix based on the last part
                if i == len(parts) - 1:
                    original_key.append("weight")
                elif i == len(parts) - 2 and parts[i+1] == "indices":
                    # For indices, keep it as is
                    original_key.append("weight")
                    original_key.append("indices")
                    i += 1
                i += 1
            else:
                # Keep other parts as is
                original_key.append(parts[i])
                i += 1
        
        return ".".join(original_key)
    
    # Default case: return the key unchanged
    return shortened_key

# Test the conversion function with specific examples
def test_conversion():
    test_keys = [
        "layers.0.attn_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.37.ffn.experts.83.w3.indices",
        "layers.12.attn.wq.weight",
        "layers.5.ffn.w1.weight",
        "embed.weight",
        "head.weight",
        "norm.weight"
    ]
    
    print("Testing reverse transformation:")
    for key in test_keys:
        original_key = convert_to_original_format(key)
        print(f"Shortened: {key} -> Original: {original_key}")

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert DeepSeek model layer names and save to a new safetensors file')
    parser.add_argument('--input', type=str, default="~/yangwang/packed_output_model/model0-mp1.safetensors",
                        help='Path to the input safetensors file')
    parser.add_argument('--output', type=str, default="./converted_model.safetensors",
                        help='Path to save the output safetensors file')
    parser.add_argument('--mapping-file', type=str, default="./transformed_keys.json",
                        help='Path to save the key mapping JSON file')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run test conversion without processing the model')
    
    args = parser.parse_args()
    
    # Run test conversion first to verify the fixes
    test_conversion()
    
    if args.test_only:
        print("\nTest only mode. Exiting.")
        exit(0)
    
    # Expand user path (e.g., ~)
    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)
    mapping_file = os.path.expanduser(args.mapping_file)
    
    print(f"\nLoading model from: {input_path}")
    
    # Load the safetensors file
    model_data = safetensors.torch.load_file(input_path)
    
    # Dictionary to store the transformed keys
    transformed_keys = {}
    # Dictionary to store the transformed tensors
    transformed_tensors = {}
    
    print(f"Found {len(model_data)} keys in the model file.")
    print("Converting keys to original format...")
    
    # Process each key and tensor
    for key, tensor in model_data.items():
        original_key = convert_to_original_format(key)
        transformed_keys[key] = original_key
        transformed_tensors[original_key] = tensor
    
    # Print a sample of the transformed keys
    print("\nSample of transformed keys (first 10):")
    for i, (k, v) in enumerate(list(transformed_keys.items())[:10]):
        print(f"{i+1}. {k} -> {v}")
    
    # Save the transformed keys to a file
    print(f"\nSaving key mappings to {mapping_file}")
    with open(mapping_file, "w") as f:
        f.write("{\n" + ",\n".join([f'    "{k}": "{v}"' for k, v in transformed_keys.items()]) + "\n}")
    
    # Save the transformed model to a new safetensors file
    print(f"Saving transformed model to {output_path}")
    safetensors.torch.save_file(transformed_tensors, output_path)
    
    print(f"Successfully saved {len(transformed_keys)} key mappings to {mapping_file}")
    print(f"Successfully saved transformed model to {output_path}")

