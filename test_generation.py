import torch
import os
import json
import random
# Import the components defined in run_mnist_vael[cite: 2]
from models.vael import MNISTPairsVAELModel
from models.vael_networks import MNISTPairsEncoder, MNISTPairsDecoder, MNISTPairsMLP
# Assuming these helper functions are in the same utility module as run_mnist_vael
from utils.mnist_utils.mnist_task_VAEL import build_model_dict, build_worlds_queries_matrix

def test_saved_model(checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the exact configuration used during training
    # The config is typically serialized in a json alongside the checkpoint in custom frameworks,
    # or you must recreate the dictionary manually if the repo only logs to a CSV[cite: 2]
    # For this script, we will mock the expected keys based on the run_mnist_vael signature:
    config = {
        'latent_dim_sym': 15,   # Updated from 10 to match checkpoint
        'latent_dim_sub': 8,   # Updated from 64 to match checkpoint
        'dropout_ENC': 0.5,     
        'dropout_DEC': 0.5,
        'dropout': 0.1
    }
    
    n_digits = 10 
    sequence_len = 2
    label_dim = n_digits * sequence_len

    # 2. Rebuild the ProbLog dependencies exactly as in the training loop[cite: 2]
    model_dict = build_model_dict(sequence_len, n_digits)
    w_q = build_worlds_queries_matrix(sequence_len, n_digits).to(device)

    # 3. Instantiate the empty architecture components[cite: 2]
    encoder = MNISTPairsEncoder(hidden_channels=64, 
                                latent_dim=config['latent_dim_sym'] + config['latent_dim_sub'],
                                dropout=config['dropout_ENC'])
    
    decoder = MNISTPairsDecoder(label_dim=label_dim, 
                                hidden_channels=64, 
                                latent_dim=config['latent_dim_sub'],
                                dropout=config['dropout_DEC'])
    
    mlp = MNISTPairsMLP(in_features=config['latent_dim_sym'], 
                        n_facts=n_digits * 2)
    
    # 4. Assemble the VAEL model shell[cite: 2]
    model = MNISTPairsVAELModel(encoder=encoder, 
                                decoder=decoder, 
                                mlp=mlp,
                                latent_dims=(config['latent_dim_sym'], config['latent_dim_sub']),
                                model_dict=model_dict, 
                                w_q=w_q, 
                                dropout=config['dropout'], 
                                is_train=False, # Set to False for inference
                                device=device).to(device)

    # 5. Load the saved weights (bypassing PyTorch 2.6 security blocks)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    last_checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(last_checkpoint['model']) # The key used by the repo is 'model'[cite: 2]
    model.eval()

    print("Model successfully rebuilt and weights loaded.")

    # 6. Execute Generation Tasks
    # The codebase already has built-in generation functions for evaluating checkpoints[cite: 2]
    from utils.mnist_utils.mnist_task_VAEL import image_generation, conditional_image_generation

    # Note: These functions likely extract run_ID to save images to disk[cite: 2]
    run_ID = "eval_run" 
    
    with torch.no_grad():
        print("\nCreating unconditional generation samples...")
        image_generation(model, run_ID, folder=checkpoint_dir)
        
        print("\nCreating conditional generation samples...")
        conditional_image_generation(model, run_ID, folder=checkpoint_dir)
        
    print(f"Done. Images saved to {checkpoint_dir}")

if __name__ == "__main__":
    # Point this to the directory containing your checkpoint.pt
    path = "/home/kauamm/Documents/IC/VAEL/experiments/vael_2digitMNIST/1/29-09-2026-13-54-08/"
    test_saved_model(path)