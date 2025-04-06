import torch
import os
import argparse

def calculate_model_size_from_dict(state_dict):
    """Calculates the total size of tensors in a state dictionary."""
    total_size_bytes = 0
    if not isinstance(state_dict, dict):
        raise TypeError("Input must be a state dictionary (dict).")
    
    for param_tensor in state_dict.values():
        if isinstance(param_tensor, torch.Tensor):
            total_size_bytes += param_tensor.numel() * param_tensor.element_size()
        # Some state dicts might contain non-tensor values (e.g., metadata)
        # We simply ignore those for size calculation.

    total_size_mb = total_size_bytes / (1024 * 1024)
    return total_size_mb

def main():
    parser = argparse.ArgumentParser(description="Calculate the size of model parameters stored in a .pth checkpoint file.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the PyTorch checkpoint (.pth) file.")
    parser.add_argument("--key", type=str, default=None,
                        help="Optional key for the state dict within the checkpoint (e.g., 'model_state_dict', 'ema_model_state_dict'). If None, tries common keys or assumes the loaded object is the state dict.")
    
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return

    print(f"Loading checkpoint: {args.checkpoint_path}")
    try:
        # Load onto CPU to avoid unnecessary GPU memory usage
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        return

    state_dict = None
    if isinstance(checkpoint, dict):
        if args.key and args.key in checkpoint:
            state_dict = checkpoint[args.key]
            print(f"Using state dict under key: '{args.key}'")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Using state dict under key: 'model_state_dict'")
        elif 'state_dict' in checkpoint: # Another common key
             state_dict = checkpoint['state_dict']
             print("Using state dict under key: 'state_dict'")
        elif 'ema_model_state_dict' in checkpoint: # Check for EMA as fallback
            state_dict = checkpoint['ema_model_state_dict']
            print("Using state dict under key: 'ema_model_state_dict'")
        else:
            # Assume the whole dictionary is the state_dict if no common keys found
            print("Warning: Could not find standard state_dict keys. Assuming the loaded dictionary *is* the state_dict.")
            state_dict = checkpoint
            
    elif isinstance(checkpoint, dict): # Check if checkpoint *is* the state dict
        print("Loaded object is a dictionary, assuming it's the state_dict.")
        state_dict = checkpoint
    else:
        print(f"Error: Loaded checkpoint is not a dictionary or a recognized state_dict format. Type: {type(checkpoint)}")
        return

    if state_dict is None:
        print("Error: Could not extract a state dictionary from the checkpoint.")
        return
        
    try:
        model_size_mb = calculate_model_size_from_dict(state_dict)
        print(f"Calculated model parameter size: {model_size_mb:.2f} MB")
    except TypeError as e:
        print(f"Error calculating size: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during size calculation: {e}")

if __name__ == "__main__":
    main()
