# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import time

import torch

from vptq.layers.vqlinear import VQuantLinear
from vptq.quantizer import NPVectorQuantizer
from vptq.utils.hessian import load_hessian, load_inv_hessian
from vptq.utils.layer_utils import find_layers, replace_layer
from vptq.vptq import VPTQ


def layer_quantizer(args, quant_args, layer, layer_idx, logger, dev, dtype, name2hessian=None, target_layers=None):
    qlinear_args = {}
    print(f"layer: {layer}")
    operators = find_layers(layer, target_layers)
    opeartor_names = [list(operators.keys())]
    logger.info(f'opeartor_names: {opeartor_names}')
    # with torch.no_grad():
    for names in opeartor_names:
        # subset: (op name, op) pairs
        subset = {n: operators[n] for n in names}
        logger.info(subset.keys())

        # for each linear
        for name in subset:
            # load Hessian
            if name2hessian is None:
                name2hessian = {
                    'self_attn.v_proj': 'qkv',
                    'self_attn.q_proj': 'qkv',
                    'self_attn.k_proj': 'qkv',
                    'self_attn.o_proj': 'o',
                    'mlp.up_proj': 'up',
                    'mlp.gate_proj': 'up',
                    'mlp.down_proj': 'down'
                }

            layer_name = f'{layer_idx}_{name2hessian[name]}.pt'
            hessian_path = f'{args.hessian_path}/{layer_name}'
            hessian, mu = load_hessian(hessian_path, logger)

            # init data
            linear = subset[name].to(dev)
            hessian.to('cpu')

            # load inv_hessian from files to reduce memory usage
            if args.inv_hessian_path is not None:
                inv_hessian_path = f'{args.inv_hessian_path}/{layer_name}'
                inv_hessian, perm, zero_idx = load_inv_hessian(inv_hessian_path, logger)
                inv_hessian.to('cpu')
                perm.to('cpu')
                zero_idx.to('cpu')
            else:
                inv_hessian = None
                perm = None
                zero_idx = None

            layer_name = f'{layer_idx}.{name}'

            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            logger.info(f'----Quantizing llama ...---- {current_time} {layer_name}')

            # init quantizer
            quantizer = NPVectorQuantizer(
                layer_name=layer_name,
                logger=logger,
                vector_lens=quant_args.vector_lens,
                num_centroids=quant_args.num_centroids,
                num_res_centroids=quant_args.num_res_centroids,
                npercent=quant_args.npercent,
                group_size=quant_args.group_size,
                group_num=quant_args.group_num,
                # enable_transpose=True,
                kmeans_mode='hessian',
                iter=quant_args.kiter,
                tol=quant_args.ktol,
                enable_norm=quant_args.enable_norm,
                norm_dim=quant_args.norm_dim,
                debug=True,
            )

            print(f'linear {linear}, weight: {linear.weight.shape, linear.weight.dtype}, scale: {linear.scale.shape, linear.scale.dtype}')
            
            if quant_args.cast_dtype == "bf16" and linear.weight.dtype == torch.float8_e4m3fn:
                from deepseek.kernel import weight_dequant 
                bfloat16_weight = weight_dequant(x=linear.weight, s=linear.scale, block_size=128)
                print(f'dequant weight from float8_e4m3fn: {bfloat16_weight.shape, bfloat16_weight.dtype}')
                linear.weight = torch.nn.Parameter(bfloat16_weight)
                print(f"dequant weight to: {linear.weight.shape, linear.weight.dtype}")
                print(f'hessian: {hessian.shape, hessian.dtype}')
             
            # init vptq algo
            _vptq = VPTQ(
                linear,
                hessian=hessian,
                inv_hessian=inv_hessian,
                perm=perm,
                quantizer=quantizer,
                zero_idx=zero_idx,
                logger=logger,
                collect_act=False,
                layer_name=layer_name,
                enable_perm=quant_args.enable_perm,
                enable_norm=quant_args.enable_norm,
                norm_dim=quant_args.norm_dim,
                debug=True
            )

            # quant by VPTQ algorithm
            _vptq.fast_vector_quant()

            quantizer = _vptq.quantizer
            perm = _vptq.quantizer.perm

            weight = linear.weight.clone().detach().to(dev)
            hessian = hessian.to(dev)

            # num_codebooks = 1
            # centroid
            # num_centroids = quantizer.num_centroids[1]
            centroids = quantizer.centroids
            indices = quantizer.indices
            indices_sign = quantizer.indices_sign
            indices_scale = quantizer.indices_scale
            
            # res centroid
            # num_res_centroids = quantizer.num_res_centroids
            res_centroids = quantizer.res_centroids
            # res_centroids = quantizer.res_centroids[1]
            res_indices = quantizer.res_indices
            # res_indices = quantizer.res_indices[1]
            res_indices_sign = quantizer.res_indices_sign

            in_features = weight.size(1)
            out_features = weight.size(0)

            # outlier_num_centroids = quantizer.num_centroids[0]
            # outlier_num_res_centroids = quantizer.num_rescluster[0]

            qlayer = VQuantLinear(
                # **outlier_kwargs,
                in_features=in_features,
                out_features=out_features,
                vector_lens=quant_args.vector_lens,
                num_centroids=quant_args.num_centroids,
                num_res_centroids=quant_args.num_res_centroids,
                # group settings
                # group_size=quantizer.group_size,
                group_num=quantizer.group_num,
                group_size=quantizer.group_size,
                outlier_size=quantizer.outlier_size,
                bias=True if linear.bias is not None else False,
                enable_norm=quant_args.enable_norm,
                norm_dim=quant_args.norm_dim,
                enable_perm=quant_args.enable_perm,
                # enable_residual=True,
                vector_quant_dim='out',
                device=dev,
                dtype=dtype,
                # indices_as_float=False,
            )

            qlinear_args[name] = qlayer.cpu().init_args

            weight_scale = _vptq.quantizer.weight_scale
            weight_bias = _vptq.quantizer.weight_bias

            qlayer.init_parameters(
                centroids=centroids,
                indices=indices,
                res_centroids=res_centroids,
                res_indices=res_indices,
                weight_scale=weight_scale,
                weight_bias=weight_bias,
                indices_sign=indices_sign,
                indices_scale=indices_scale,
                res_indices_sign=res_indices_sign,
                bias=linear.bias,
                perm=perm,
                dtype=dtype,
            )

            qlayer.to(dev)

            # replace layer with qlinear
            replace_layer(layer, name, qlayer)

            # del quantizer
            # del _vptq
            # del qlayer
            # del qweight
            torch.cuda.empty_cache()

    return layer, qlinear_args
