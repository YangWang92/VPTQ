import argparse
import pickle
import json
import torch

class TorchDtypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        # Handle any other custom types if needed
        return super().default(obj)

def convert_dtypes_to_str(obj):
    """Recursively convert torch.dtype objects to strings in nested dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: convert_dtypes_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dtypes_to_str(item) for item in obj]
    elif isinstance(obj, torch.dtype):
        return str(obj)
    else:
        return obj

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_vptq_config', type=str, required=True)
    parser.add_argument('--input_deepseek_config', type=str, required=True)
    parser.add_argument('--output_config', type=str, required=True)
    args = parser.parse_args()

    with open(args.input_vptq_config, 'rb') as f:
        vptq_config = pickle.load(f)

    with open(args.input_deepseek_config, 'rb') as f:
        deepseek_config = json.load(f)

    # print(vptq_config)
    # print(deepseek_config)

    # delete quantization in deepseek_config
    del deepseek_config['quantization_config']

    # Convert torch.dtype objects to strings in config_for_layers
    config_for_layers = convert_dtypes_to_str(vptq_config['config_for_layers'])
    
    new_config_for_layers = {}

    # update config_for_layers
    for key, value in config_for_layers.items():
        print(f'update {key}')
        value['enable_perm'] = False
        value['is_indice_packed'] = True
        
        # New approach: convert to shortened format
        # First, create the original format key
        original_format_key = 'layers.' + key
        # Then convert to the shortened format
        new_key = convert_to_original_format(original_format_key)
        
        new_config_for_layers[new_key] = value
        print(f'Converted: {original_format_key} -> {new_key}')
        print('----')
    
    deepseek_repo_config = deepseek_config.copy()
     
    deepseek_config['quantization_config'] = {
        'config_for_layers': new_config_for_layers,
        'quant_method': 'vptq'
    } 
    with open(args.output_config, 'w') as f:
        json.dump(deepseek_config, f, cls=TorchDtypeEncoder, indent=2)
    
    with open(f'repo_{args.output_config}', 'w') as f:
        json.dump(deepseek_repo_config, f, cls=TorchDtypeEncoder, indent=2)
     
if __name__ == '__main__':
    main()