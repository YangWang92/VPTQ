import torch
from transformers import LlamaForCausalLM
from vptq.quantizer import QuantizationArguments
from vptq.weight_quantizer import weight_quantizer 
from vptq.utils.hessian import load_hessian 
from argparse import ArgumentParser
import logging
def load_model():
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", 
                                          torch_dtype=torch.bfloat16, 
                                          device_map='auto')	
    return model

def load_weight(model):
    weight = model.model.layers[0].mlp.down_proj.weight
    # detach weight
    weight = weight.detach()
    return weight

def build_quant_args():
    quant_args = QuantizationArguments(
        vector_lens=[-1, 8], # vector length = 8
        num_centroids=[-1, 65536], # number of centroids = 65536, log2(65536) = 16 / 8 = 2 bits
        num_res_centroids=[-1, 256], # number of residual centroids = 256, log2(256) = 8 / 8 = 1 bit
        npercent=0, # no outliers
        group_num=1, # single group
        kiter=10, # kmeans iterations = 10
        ktol=1e-5, # kmeans tolerance = 1e-5m
        enable_norm=True,
        norm_dim=0,
    )
    return quant_args

def build_logger():
    logger = logging.getLogger('weight_quantizer')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def test_weight_quantizer(args):
    model = load_model()
    weight = load_weight(model)
    hessian, _ = load_hessian(args.hessian_path)
    print(f'weight shape: {weight.shape}')
    print(f'hessian shape: {hessian.shape}')
     
    quant_args = build_quant_args()
    logger = build_logger()
    dev = torch.device('cuda:0')
 
    vqlinear, vqlinear_args = weight_quantizer(quant_args=quant_args, 
                  weight=weight, 
                  hessian=hessian, 
                  bias=None, 
                  logger=logger, 
                  dev=dev, 
                  dtype=torch.bfloat16)

    print(f'vqlinear: {vqlinear}')
    print(f'vqlinear shape: {vqlinear.dequant().shape}')
    qweight = vqlinear.dequant()
    # get mean abs error
    mean_abs_error = torch.mean(torch.abs(weight - qweight))
    print(f'mean abs error: {mean_abs_error}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--hessian_path', type=str, default='Hessians-Llama-31-8B-Instruct-6144-8k/0_down.pt')
    args = parser.parse_args()
    test_weight_quantizer(args)
