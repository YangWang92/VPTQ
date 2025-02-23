import torch
import vptq
import argparse
from deepseek.model import Transformer, ModelArgs
import json
import os
import time
from vptq.utils.layer_utils import find_layers, replace_layer

def load_model(ckpt_path: str, config_path: str, dry_run: bool) -> Transformer:
    with open(config_path) as f:
        model_args = ModelArgs(**json.load(f))
    with torch.device("cpu"):  # Always load model to CPU first
        model = Transformer(model_args)
        # Always load distributed model files
        if not dry_run:
            load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
            print(f"load_model time: {time.time() - start_time:.2f}s")
        else:
            print(f"dry run model load")
    return model       

def create_vqlinear(layer_qlinear_args, layer_state_dict, dtype):
    vqlinear = VQuantLinear(
        **layer_qlinear_args[layer_idx][name],
        dtype=dtype,
    )
    vqlinear.load_state_dict(layer_state_dict)
    return vqlinear

def get_checkpoint_layer_list(path: str, num_layers: int=60) -> list[tuple[int, str, str]]:
    layer_list = {}
    for i in range(num_layers):
        layer_list[i] = (i, os.path.join(path, f"qlinear_args_{i}.pt"), os.path.join(path, f"qlinear_layer_state_{i}.pt"))
    return layer_list

def get_quantized_deepseek(model, layer_list, dtype=torch.bfloat16):
    layers = model.layers
    
    for layer_idx, layer_state_dict in layer_list.items():
        # print(f'load quantized layer {layer_idx}')
        # print(f'layer_state_dict: {layer_state_dict.keys()}')
        layer = layers[layer_idx]
        ops = find_layers(layer)

        for name, op in ops.items():
            # init qlinear
            qlayer = VQuantLinear(
                **layer_qlinear_args[layer_idx][name],
                dtype=dtype,
            )
            module_name = name.split('.')[-1]
            replace_layer(layer, module_name, qlayer)

        # convert dtype
        # print(f'default dtype: {dtype}')
        for param_name, param in layer_state_dict.items():
            if layer_state_dict[param_name].dtype not in [
                dtype, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint64, torch.uint32, torch.uint16,
                torch.uint8, torch.bool
            ]:
                layer_state_dict[param_name] = layer_state_dict[param_name].to(dtype)

        layers[layer_idx].load_state_dict(layer_state_dict)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=True)
    parser.add_argument("--input_config", type=str, required=True)
    parser.add_argument("--input_quantized_ckpt", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = load_model(args.input_model, args.input_config, args.dry_run)
    
    print(model)

    # list_files(args.input_quantized_ckpt)
    layer_list = get_checkpoint_layer_list(args.input_quantized_ckpt)
    print(layer_list)

if __name__ == "__main__":
    main()
