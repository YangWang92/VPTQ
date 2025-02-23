import torch
import argparse
from deepseek.model import Transformer, ModelArgs
import json
import os
import time
from vptq.utils.layer_utils import find_layers, replace_layer
from vptq.layers.vqlinear import VQuantLinear
from deepseek.model import ColumnParallelLinear, RowParallelLinear, Linear
from safetensors.torch import load_model


def load_model_from_safetensors(ckpt_path: str, config_path: str, dry_run: bool, world_size: int=1) -> Transformer:
    with open(config_path) as f:
        model_args = ModelArgs(**json.load(f))
    with torch.device("cpu"):  # Always load model to CPU first
        model = Transformer(model_args)
        # Always load distributed model files
        if not dry_run:
            load_model(model, os.path.join(ckpt_path, f"model-mp{world_size}.safetensors"))
            print(f"load_model time: {time.time() - os.startfile:.2f}s")
        else:
            print(f"dry run model load")
    return model       


def create_vqlinear(layer_qlinear_args, layer_state_dict, dtype=torch.bfloat16):
    vqlinear = VQuantLinear(
        **layer_qlinear_args,
        dtype=dtype,
    )
    vqlinear.load_state_dict(layer_state_dict)
    return vqlinear

def get_checkpoint_layer_list(path: str, num_layers: int=60) -> list[tuple[int, str, str]]:
    layer_list = [] 
    for i in range(num_layers):
        layer_list.append((i, os.path.join(path, f"qlinear_args_{i}.pt"), os.path.join(path, f"qlinear_layer_state_{i}.pt")))
    return layer_list

def get_quantized_deepseek(model, path, dtype=torch.bfloat16):
    num_layers = len(model.layers)
    layers = model.layers
    target_layers = [ColumnParallelLinear, RowParallelLinear, Linear]
    layer_list = get_checkpoint_layer_list(path, num_layers)
    
    for (layer_idx, layer_qlinear_args, layer_state_dict) in layer_list:
        print(f'--------------------------------') 
        
        print(f'layer_idx: {layer_idx}')
        layer = layers[layer_idx]
        
        layer_qlinear_args = torch.load(layer_qlinear_args)
        layer_state_dict = torch.load(layer_state_dict)
        
        ops = find_layers(layer, target_layers)
        
        print(f'ops from original model: {ops.keys()}')
        print(f'--------------------------------') 
        for module_name, op in ops.items():
            # init qlinear
            print(f'init module_name: {module_name}')
            qlinear = VQuantLinear(
                **layer_qlinear_args[module_name],
                dtype=dtype,
            )
            replace_layer(layer, module_name, qlinear)
        
        layer.load_state_dict(layer_state_dict[module_name])
       
        del layer_qlinear_args
        del layer_state_dict
    print(f'--------------------------------')
    print(f'quantized model: {model}')
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=True)
    parser.add_argument("--input_config", type=str, required=True)
    parser.add_argument("--input_quantized_ckpt", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = load_model_from_safetensors(args.input_model, args.input_config, args.dry_run)
    print(model)
    
    # get quantized model
    quantized_model = get_quantized_deepseek(model, args.input_quantized_ckpt)
    print(quantized_model)

if __name__ == "__main__":
    main()
