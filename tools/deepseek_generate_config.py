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
        
        new_key = 'model.layers.' + key
        new_config_for_layers[new_key] = value
        print('----')
    
    deepseek_config['quantization_config'] = {
        'config_for_layers': new_config_for_layers,
        'quant_method': 'vptq'
    } 
    with open(args.output_config, 'w') as f:
        json.dump(deepseek_config, f, cls=TorchDtypeEncoder, indent=2)
        
if __name__ == '__main__':
    main()