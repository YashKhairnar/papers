import torch
import os

def prepare_for_hf():
    input_file = "best_a100_model_100.pt"
    output_file = "deploy_model_fp16.pt"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading {input_file} (2.5GB typically)...")
    checkpoint = torch.load(input_file, map_location='cpu')
    
    # 1. Strip redundant training states
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 2. Perform FP16 quantization (reduces size by 50%)
    print("Compressing weights to FP16...")
    fp16_state_dict = {k: v.half() for k, v in state_dict.items()}
    
    deploy_dict = {
        'model_state_dict': fp16_state_dict
    }
    
    print(f"Saving ultralight model to {output_file}...")
    torch.save(deploy_dict, output_file)
    
    old_size = os.path.getsize(input_file) / (1024**3)
    new_size = os.path.getsize(output_file) / (1024**3)
    
    print("\n" + "="*30)
    print(f"Deployment Model Optimized (FP16)")
    print(f"Original: {old_size:.2f} GB")
    print(f"Optimized: {new_size:.3f} GB")
    print(f"Status: {'✅ Fits in 1GB Space (Safe for 2GB RAM)' if new_size < 1.0 else '❌ Too Large'}")
    print("="*30)

if __name__ == "__main__":
    prepare_for_hf()
