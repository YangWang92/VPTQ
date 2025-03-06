import argparse
import math
import os
from site import makepath

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

import vptq
from vptq.utils.pack import dtype_convert, pack_index, unpack_index_tensor

def load_model_from_safetensors(ckpt_path: str, config_path: str, dry_run: bool, world_size: int=1) -> Transformer:
    with open(config_path) as f:
        model_args = ModelArgs(**json.load(f))
    with torch.device("cpu"):  # Always load model to CPU first
        model = Transformer(model_args)
        # Always load distributed model files
        if not dry_run:
            load_model(model, os.path.join(ckpt_path, f"model0-mp{world_size}.safetensors"))
        else:
            print(f"dry run model load")
    return model       


def get_checkpoint_layer_list(path: str, num_layers: int=60) -> list[tuple[int, str, str]]:
    layer_list = []
    for i in range(num_layers):
        layer_list.append((i, os.path.join(path, f"qlinear_args_{i}.pt"), os.path.join(path, f"qlinear_layer_state_{i}.pt")))
    return layer_list


def get_quantized_deepseek(model, path, dtype=torch.bfloat16, enable_pack: bool=False, enable_absorb_perm: bool=False):
    quantization_config = {
        "quant_method": "vptq",
        "config_for_layers": {}
    }
    num_layers = len(model.layers)
    layers = model.layers
    target_layers = [ColumnParallelLinear, RowParallelLinear, Linear]
    layer_list = get_checkpoint_layer_list(path, num_layers)
    
    for (layer_idx, layer_qlinear_args_path, layer_state_dict_path) in layer_list:
        print(f'handle layer {layer_idx}')
        layer = layers[layer_idx]
         
        layer_qlinear_args = torch.load(layer_qlinear_args_path, weights_only=False)
        layer_state_dict = torch.load(layer_state_dict_path, weights_only=False)
        
        layer = layer.to('cuda')
        
        ops = find_layers(layer, target_layers)

        for module_name, op in ops.items():
            # init qlinear
            qlinear = VQuantLinear(
                **layer_qlinear_args[module_name],
                dtype=dtype
            )
            qlinear = qlinear.to('cuda')
            replace_layer(layer, module_name, qlinear)
            # write to quant_config
            quantization_config['config_for_layers'][f'{layer_idx}.{module_name}'] = {
                **layer_qlinear_args[module_name],
                "dtype": dtype
            }
        layer.load_state_dict(layer_state_dict)
        print(f'replace layer {layer_idx}')
        
        # pack and absorb perm
        if enable_pack or enable_absorb_perm:
            ops = find_layers(layer, [VQuantLinear]) 
            for module_name, op in ops.items():
                op = process_linear(op, enable_pack, enable_absorb_perm)
                if enable_absorb_perm:
                    print(f'absorb perm layer: {layer_idx}')
                if enable_pack:
                    print(f'pack indices layer: {layer_idx}')
                replace_layer(layer, module_name, op)
        
        layer = layer.to('cpu')
        del layer_qlinear_args
        del layer_state_dict
        print(f'--------------------------------')
        
    print(f'quantized model: {model}')
    return model, quantization_config


def save_model(model: Transformer, save_path: str, world_size: int = 1):
    os.makedirs(save_path, exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), f"{save_path}/model0-mp{world_size}.safetensors")
    print(f"Model saved to {save_path}")


def process_linear(linear, quant_config, enable_pack: bool=False, enable_absorb_perm: bool=False):
    if not hasattr(linear, 'enable_perm') or not linear.enable_perm:
        return linear

    print(f'linear.enable_perm: {linear.enable_perm}, perm dtype: {linear.perm.dtype}')
    # Get inverse permutation
    if linear.is_indice_packed:
        invert_perm = torch.argsort(linear.perm.view(torch.uint16).to(torch.int64))
    else:
        invert_perm = torch.argsort(linear.perm)

    index_bits = int(math.log2(linear.num_centroids))
    index_res_bits = int(math.log2(linear.num_res_centroids)) if linear.enable_residual else 0
    print(f'index_bits: {index_bits}, index_res_bits: {index_res_bits}')
    print(f'input linear.indices shape: {linear.indices.shape}')
    print(f'linear.group_size: {linear.group_size}')

    # Rotate indices based on permutation
    if linear.is_indice_packed:
        # Unpack indices
        indices, res_indices = linear.unpack_index_tensor(
            pack_tensor=linear.indices,
            index_bits=index_bits,
            num_elements=linear.group_size,
            res_bits=index_res_bits,
            num_res_elements=linear.group_size,
            index_dtype=torch.uint16,
        )

        print(f'unpack indices shape: {indices.shape}, dtype: {indices.dtype}')
        print(f'unpack res_indices shape: {res_indices.shape}, dtype: {res_indices.dtype}')

        # Apply permutation
        indices = indices[..., invert_perm]
        if linear.enable_residual:
            res_indices = res_indices[..., invert_perm]

        indices = dtype_convert(indices, torch.int64, torch.uint16, torch.uint16)
        if linear.enable_residual:
            res_indices = dtype_convert(res_indices, torch.int64, torch.uint16, torch.uint16)

        print(f'perm indices shape: {indices.shape}')
        print(f'perm res_indices shape: {res_indices.shape}')

        # Pack indices back
        packed_indices = pack_index(
            indice=indices,
            index_bits=index_bits,
            res_indice=res_indices if linear.enable_residual else None,
            res_bits=index_res_bits if linear.enable_residual else 0,
            index_dtype=torch.uint16
        )

        # work around for packed indices shape
        print(f'packed_indices shape: {packed_indices.shape}')

        # Ensure packed shape matches original
        if packed_indices.shape != linear.indices.shape:
            raise ValueError(f"Packed shape {packed_indices.shape} doesn't match original shape {linear.indices.shape}")

        linear.indices.data = packed_indices
        print(f'repacked linear.indices shape: {linear.indices.shape}')
        print('-------')
    else:
        indices = linear.indices
        indices = indices[..., invert_perm]
        linear.indices.data = indices

        if linear.enable_residual:
            res_indices = linear.res_indices
            res_indices = res_indices[..., invert_perm]
            linear.res_indices.data = res_indices

        if enable_pack:
            # check it carefullly
            indices = dtype_convert(indices, torch.uint16, torch.uint16, torch.uint16)
            if linear.enable_residual:
                res_indices = dtype_convert(res_indices, torch.uint16, torch.uint16, torch.uint16)

            # Pack indices back
            packed_indices = pack_index(
                indice=indices,
                index_bits=index_bits,
                res_indice=res_indices if linear.enable_residual else None,
                res_bits=index_res_bits if linear.enable_residual else 0,
                index_dtype=torch.uint16
            )

            # work around for packed indices shape
            print(f'packed_indices shape: {packed_indices.shape},'
                  f'enable_residual: {linear.enable_residual},'
                  f'res_indices shape: {res_indices.shape if linear.enable_residual else None}')

            linear.indices.data = packed_indices
            if linear.enable_residual:
                linear.res_indices = None
            print(f'repacked linear.indices shape: {linear.indices.shape}')
            print('-------')

    # Handle weight scale and bias if enable_norm is True
    if linear.enable_norm:
        if hasattr(linear, 'norm_dim') is False:
            linear.norm_dim = 0

    # Disable permutation
    linear.enable_perm = False
    linear.perm = None
    return linear

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=True)
    parser.add_argument("--input_config", type=str, required=True)
    parser.add_argument("--layer_args_path", type=str, default="")
    parser.add_argument("--input_quantized_ckpt", type=str, required=True)

    # enable pack and absorb perm
    parser.add_argument("--enable_pack", action="store_true")
    parser.add_argument("--enable_absorb_perm", action="store_true")

    # write to file
    parser.add_argument("--output_model", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()   
    torch.set_num_threads(os.cpu_count())
    
    model = load_model_from_safetensors(args.input_model, args.input_config, args.dry_run)
    print(model)
    
    # get quantized model
    quantized_model, quantization_config = get_quantized_deepseek(model, 
                                             args.input_quantized_ckpt, 
                                             dtype=torch.bfloat16,
                                             enable_pack=args.enable_pack, 
                                             enable_absorb_perm=args.enable_absorb_perm)
    print(quantized_model)

    # save quantized model
    save_model(quantized_model, args.output_model)

    # save quant_config
    # with open(args.output_model + 'quantization_config.json', 'w') as f:
    #     json.dump(quantization_config, f)
    import pickle
    with open(args.output_model + 'quantization_config.pkl', 'wb') as f:
        pickle.dump(quantization_config, f)

if __name__ == "__main__":
    main()

