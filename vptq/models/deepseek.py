# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Model from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/model/llama.py
# Evluation from https://github.com/IST-DASLab/gptq/blob/main/eval.py

import os
from pkgutil import get_data
from sys import argv
from sysconfig import get_path
import time
import json

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm

from vptq.layers.vqlinear import VQuantLinear
from vptq.quantize_executer import quantize_executer
from vptq.utils.layer_utils import find_layers, replace_layer
from deepseek.model import Transformer, ModelArgs
from safetensors.torch import load_model
from deepseek.model import ColumnParallelLinear, RowParallelLinear, Linear

# get deepseek model
def get_deepseek(model_name, config_path, enable_offload=False):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    # load config
    with open(config_path) as f:
        model_args = ModelArgs(**json.load(f))
    
    print(f'deepseek config: {model_args}')

    # if seqlen is not None:
    #     model.seqlen = seqlen
        
    if enable_offload:
        # load layer list
        # layers = {}
        layers = []
        layer_files = sorted([f for f in os.listdir(model_name) if f.startswith('layer') and f.endswith('.pt')])
        for layer_file in layer_files:
            layer_idx = int(layer_file.split('_')[-1].split('.')[0])
            layer_path = os.path.join(model_name, layer_file)
            # layers[layer_idx] = layer_path
            layers.append(layer_path)
        
        print(f'get offload layers: {layers}')
        # load model
        class OffloadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = None
                self.layers = layers
                self.model_args = model_args
            def forward(self, x):
                return x
        print(f"offload model loaded")
        model = OffloadModel()
    else:
        with torch.device("cpu"):  # Always load model to CPU first
            model = Transformer(model_args)
            # Always load distributed model files
            start_time = time.time()
            path = os.path.join(model_name, f"model0-mp1.safetensors")
            print(f"load_model path: {path}")
            load_model(model, path)
            print(f"load_model time: {time.time() - start_time:.2f}s")
    
    # debug
    print(f"model: {model}")
    return model


# quant deepseek model
def quant_deepseek(model, args, quant_args, dev='cuda'):
    # model.model.required_grad = False
    print('Starting VPTQ...')

    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    
    model = model.cpu()

    # multiple gpus VPTQ
    quantizers = {}
    layers = model.layers

    print(f'----quantization start ...---- {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}')

    # calculate task allocation
    total_layers = len(layers)
    num_gpus = min(args.num_gpus, total_layers)

    base_layers_per_gpu = total_layers // num_gpus
    remaining_layers = total_layers % num_gpus

    tasks = []
    current_layer_idx = 0

    # Distribute tasks to GPUs
    for gpu_idx in range(num_gpus):
        current_gpu_tasks = []

        # Calculate how many layers this GPU should handle
        layers_for_this_gpu = base_layers_per_gpu
        if gpu_idx < remaining_layers:
            layers_for_this_gpu += 1

        # Assign layers to this GPU
        for _ in range(layers_for_this_gpu):
            current_gpu_tasks.append((current_layer_idx, layers[current_layer_idx]))
            current_layer_idx += 1

        tasks.append(current_gpu_tasks)

    # print task allocation
    for gpu_idx in range(len(tasks)):
        task = [layer_idx for layer_idx, _ in tasks[gpu_idx]]
        print(f'gpu {gpu_idx} tasks: {task}')

    # init multiprocessing
    processes = []
    mq_manager = mp.get_context('spawn').Manager()
    input_queues = mq_manager.Queue()
    output_queues = mq_manager.Queue()

    # rename to hessian
    name2hessian = {
        'attn.wq_a': 'attn.wq_a',
        'attn.wq_b': 'attn.wq_b',
        'attn.wkv_a': 'attn.wkv_a',
        'attn.wkv_b': 'attn.wkv_b',
        'attn.wo': 'attn.wo',
        'ffn.w1': 'ffn.w1',
        'ffn.w2': 'ffn.w2',
        'ffn.w3': 'ffn.w1',
    }

    # init MoE name to hessian, shared_experts=1 for DeepSeek-V3
    if model.model_args.n_shared_experts == 1:
        name2hessian[f'ffn.shared_experts.w1'] = f'ffn.shared_experts.w1'
        name2hessian[f'ffn.shared_experts.w2'] = f'ffn.shared_experts.w2'
        name2hessian[f'ffn.shared_experts.w3'] = f'ffn.shared_experts.w1'
    else:
        for i in range(model.model_args.n_shared_experts):
            name2hessian[f'ffn.shared_experts.{i}.w1'] = f'ffn.shared_experts.{i}.w1'
            name2hessian[f'ffn.shared_experts.{i}.w2'] = f'ffn.shared_experts.{i}.w2'
            name2hessian[f'ffn.shared_experts.{i}.w3'] = f'ffn.shared_experts.{i}.w1'
    
    for i in range(model.model_args.n_routed_experts):
        name2hessian[f'ffn.experts.{i}.w1'] = f'ffn.experts.{i}.w1'
        name2hessian[f'ffn.experts.{i}.w2'] = f'ffn.experts.{i}.w2'
        name2hessian[f'ffn.experts.{i}.w3'] = f'ffn.experts.{i}.w1'
    
    target_layers = [ColumnParallelLinear, RowParallelLinear, Linear]

    if args.num_gpus == 1:
        layer_state_dicts, layer_qlinear_args = quantize_executer(0, tasks[0], args, 
                                                                  quant_args, None, None, 
                                                                  name2hessian, dev=None, 
                                                                  target_layers=target_layers)
    else:
        for gpu_idx in range(args.num_gpus):
            # we have to set CUDA_VISIBLE_DEVICES here
            # cuml only supports to run on GPU:0
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            dev = f'cuda:0'
            p = mp.Process(
                target=quantize_executer,
                args=(
                    gpu_idx,
                    tasks[gpu_idx],
                    args,
                    quant_args,
                    input_queues,
                    output_queues,
                    name2hessian,
                    target_layers,
                )
            )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print(f'----quantization done ...---- {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}')

    # init quantized model
    model_name = 'deepseek' 

    if args.num_gpus > 1:
        layer_state_dicts = {}
        layer_qlinear_args = {}

        # load save qlinear from files to avoid memory overflow
        if args.save_qlinear:
            for layer_idx in range(len(layers)):
                # load to cpu
                layer_state_dicts[layer_idx] = torch.load(
                    f'{args.output_dir}/qlinear_layer_state_{layer_idx}.pt', map_location='cpu',
                    weights_only=False
                )
                # bypass KeyError: torch.uint16
                for key, value in layer_state_dicts[layer_idx].items():
                    if "indices" in key:
                        layer_state_dicts[layer_idx][key] = value.view(torch.uint16)
                layer_qlinear_args[layer_idx] = torch.load(
                    f'{args.output_dir}/qlinear_args_{layer_idx}.pt', map_location='cpu',
                    weights_only=False
                )
        else:
            while not output_queues.empty():
                (gpu_id, layer_idx, _layer_state_dict, _layer_qlinear_args) = output_queues.get()
                layer_state_dicts[layer_idx] = _layer_state_dict
                layer_qlinear_args[layer_idx] = _layer_qlinear_args
                print(f'gpu {gpu_id} layer {layer_idx} quantized')

    # check if all layers are quantized
    if len(layer_state_dicts) != len(layers):
        print('Error: not all layers are quantized')
        exit(1)

    qmodel = get_quantized_deepseek(model_name, args.seq_len, layer_state_dicts, layer_qlinear_args)

    model = qmodel

    print(f'qmodel: {model}')

    torch.cuda.empty_cache()
    return model, quantizers


def get_quantized_deepseek(model_name, seqlen, layer_state_dicts, layer_qlinear_args):

    # print(f'layer_state_dicts: {layer_state_dicts.keys()}')
    model = get_data(model_name, seqlen=seqlen)
    dtype = next(iter(model.parameters())).dtype
    layers = model.layers

    for layer_idx, layer_state_dict in layer_state_dicts.items():
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

    return model


@torch.no_grad()
def eval_llama(model, testenc, dev):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f'----Evaluating llama ...---- {current_time}')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # fix for llama-3.1
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            if hasattr(model.model, 'rotary_emb'):
                cache['rotary_emb'] = model.model.rotary_emb(x=inp, position_ids=kwargs['position_ids'])
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(input_ids=batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    # get position embeddings from the model's rotary embeddings
    # for the latest huggingface transformers
    position_embeddings = model.model.rotary_emb(outs, position_ids)

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), 
                          attention_mask=attention_mask, 
                          position_ids=position_ids,
                          position_embeddings=position_embeddings)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()
