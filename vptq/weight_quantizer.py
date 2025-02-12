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


def weight_quantizer(quant_args, weight, hessian, bias, logger, dev, dtype):
    # move weight to device
    weight = weight.to(dev)
    hessian.to('cpu')
    
    perm = None
    zero_idx = None
    
    layer_name = 'linear'

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
        kmeans_mode='hessian',
        iter=quant_args.kiter,
        tol=quant_args.ktol,
        enable_norm=quant_args.enable_norm,
        norm_dim=quant_args.norm_dim,
        debug=True,
    )

    # init vptq algo
    _vptq = VPTQ(
        weight=weight,
        hessian=hessian,
        inv_hessian=None,
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

    weight = weight.to(dev)
    hessian = hessian.to(dev)

    centroids = quantizer.centroids
    indices = quantizer.indices
    indices_sign = quantizer.indices_sign
    indices_scale = quantizer.indices_scale
    
    res_centroids = quantizer.res_centroids
    res_indices = quantizer.res_indices
    res_indices_sign = quantizer.res_indices_sign

    in_features = weight.size(1)
    out_features = weight.size(0)

    qlinear = VQuantLinear(
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
        bias=True if bias is not None else False,
        enable_norm=quant_args.enable_norm,
        norm_dim=quant_args.norm_dim,
        enable_perm=quant_args.enable_perm,
        # enable_residual=True,
        vector_quant_dim='out',
        device=dev,
        dtype=dtype,
        # indices_as_float=False,
    )

    qlinear_args = qlinear.cpu().init_args

    weight_scale = _vptq.quantizer.weight_scale
    weight_bias = _vptq.quantizer.weight_bias

    qlinear.init_parameters(
        centroids=centroids,
        indices=indices,
        res_centroids=res_centroids,
        res_indices=res_indices,
        weight_scale=weight_scale,
        weight_bias=weight_bias,
        indices_sign=indices_sign,
        indices_scale=indices_scale,
        res_indices_sign=res_indices_sign,
        bias=bias,
        perm=perm,
        dtype=dtype,
    )

    qlinear.to(dev)

    torch.cuda.empty_cache()

    return qlinear, qlinear_args
