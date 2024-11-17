import torch
from vptq import AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
import os
from datetime import datetime


def profile_model(model_name, model, input_len, generate_len, save_dir):
    device = next(model.parameters()).device
    model.eval()
    
    # Create dummy input
    input_ids = torch.randint(0, 32000, (1, input_len), device=device)
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    profile_dir = os.path.join(
        save_dir, 
        f"{model_name.split('/')[-1]}_in{input_len}_gen{generate_len}_{timestamp}"
    )
    os.makedirs(profile_dir, exist_ok=True)

    # Create tensorboard trace handler
    tb_trace_handler = tensorboard_trace_handler(profile_dir)

    # Define profiler schedule
    prof_schedule = schedule(
        wait=1,
        warmup=1,
        active=5,
        repeat=2
    )

    prof = profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=prof_schedule,
        on_trace_ready=tb_trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        with_flops=True
    )
    
        
    prof.start()
    for step in range(2):
        prof.step()
        with record_function("inference"):
            outputs = model(input_ids)
            # print(outputs)
        
        with record_function("generation"):
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=generate_len,
                pad_token_id=2,
                do_sample=True
            )
    prof.stop()
    
    return profile_dir


def main():
    # Configuration
    save_dir = "profiling_results"
    input_seq_lens = [32, 256, 1024, 2048, 4096]
    generate_lens = [32, 256, 1024, 2048, 4096]
    
    models = [
        {
            "name": "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v8-k65536-65536-woft",
            "bits": "4 bits"
        },
        {
            "name": "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v8-k65536-256-woft",
            "bits": "3 bits"
        },
        {
            "name": "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft",
            "bits": "2.3 bits"
        },
    ]

    for model_config in models:
        model_name = model_config["name"]
        print(f"\nProfiling {model_name}")
        
        for input_len in input_seq_lens:
            for generate_len in generate_lens:
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
                # model.to('cuda:0')
                model.eval()
        
                print(f"\nInput length: {input_len}, Generate length: {generate_len}")
                
                save_path = profile_model(
                    model_name, 
                    model, 
                    input_len, 
                    generate_len, 
                    save_dir
                )
                
                print(f"Profile results saved to: {save_path}")        
        
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()