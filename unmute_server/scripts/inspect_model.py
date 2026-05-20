import torch
from safetensors import safe_open
model_path = "/mnt/matylda4/udupa/exps/full_duplex/moshi-finetune/checkpoints/delay3_en_zh-v6/checkpoints/best_003800/consolidated/consolidated.safetensors"
ref_path = "/mnt/matylda4/udupa/hugging-face/hub/models--kyutai--stt-1b-en_fr-candle/blobs/b9e97c53229dce728d65c76bfa892f7b563c69d671899f0ebc6518582dddec6f"
output_path = "/mnt/matylda4/udupa/exps/full_duplex/moshi-finetune/checkpoints/delay3_en_zh-v6/checkpoints/best_003800/consolidated/consolidated_renamed.safetensors"

from safetensors import safe_open
from safetensors.torch import save_file
import torch

def list_model_param_names(model_path, ref_path):
    """List parameter names in the reference model"""
    print("Loading reference model...")
    with safe_open(ref_path, framework="pt") as f:
        keys = list(f.keys())
    with safe_open(model_path, framework="pt") as f:
        model_keys = list(f.keys())
    
    ## keys of ref_path not in model_path

    missing_keys = set(keys) - set(model_keys)
    if missing_keys:
        print("Keys in reference model but missing in target model:")
        for key in missing_keys:
            print(f"  - {key}")
    
    return keys

# list_model_param_names(model_path, ref_path)


def convert_model(input_path, output_path):
    """Convert model tensor names to match Rust inference expectations"""
    
    print("Loading model...")
    tensors = {}
    
    with safe_open(input_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    print(f"Loaded {len(tensors)} tensors")
    
    # Create mapping for renaming
    renamed_tensors = {}
    rename_count = 0
    
    for old_key, tensor in tensors.items():
        new_key = old_key
        
        # Rename in_projs.0.weight -> in_proj_weight
        if ".in_projs.0.weight" in old_key:
            new_key = old_key.replace(".in_projs.0.weight", ".in_proj_weight")
            rename_count += 1
            print(f"  Renaming: {old_key} -> {new_key}")
        
        # Rename out_projs.0.weight -> out_proj.weight
        elif ".out_projs.0.weight" in old_key:
            new_key = old_key.replace(".out_projs.0.weight", ".out_proj.weight")
            rename_count += 1
            print(f"  Renaming: {old_key} -> {new_key}")
        
        renamed_tensors[new_key] = tensor
    
    # Add missing extra_heads for pause prediction (dummy/random init)
    # These are 4 linear layers: nn.Linear(6, extra_heads_dim, bias=False)
    # Based on the model definition: num_heads=4, dim=6
    extra_heads_num = 4
    input_dim = 6
    output_dim = 2048  # Assuming same as input_dim, adjust if needed
    
    print(f"\nAdding {extra_heads_num} extra_heads for pause prediction...")
    for i in range(extra_heads_num):
        key = f"extra_heads.{i}.weight"
        if key not in renamed_tensors:
            # Initialize with small random values (xavier uniform)
            weight = torch.empty(input_dim, output_dim)
            torch.nn.init.xavier_uniform_(weight)
            renamed_tensors[key] = weight
            print(f"  Added: {key} with shape {weight.shape}")
    
    print(f"\nRenamed {rename_count} tensors")
    print(f"Saving converted model to {output_path}...")
    
    # Save the converted model
    save_file(renamed_tensors, output_path)
    
    print("✅ Conversion complete!")
    
    # Verify the conversion
    print("\nVerifying conversion...")
    with safe_open(output_path, framework="pt") as f:
        converted_keys = set(f.keys())
        
        # Check if expected keys exist
        expected_keys = [
            "transformer.layers.0.self_attn.in_proj_weight",
            "transformer.layers.0.self_attn.out_proj.weight",
        ]
        
        all_found = True
        for key in expected_keys:
            if key in converted_keys:
                print(f"  ✓ {key}")
            else:
                print(f"  ✗ {key} NOT FOUND")
                all_found = False
        
        # Check no old keys remain
        old_pattern_found = False
        for key in converted_keys:
            if ".in_projs." in key or ".out_projs." in key:
                print(f"  ⚠️  Old pattern still exists: {key}")
                old_pattern_found = True
        
        if all_found and not old_pattern_found:
            print("\n✅ Verification passed! Model is ready for Rust inference.")
        else:
            print("\n⚠️  Verification failed. Please check the output.")

# Run the conversion
convert_model(model_path, output_path)