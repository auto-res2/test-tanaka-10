"""
Main script for running the FaFT experiments.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.faft_config import *
from src.train import FaFTModel, ConditionedFaFTModel, train_model
from src.evaluate import evaluate_resolution_extrapolation, evaluate_factorized_conditioning
from src.preprocess import create_dummy_dataset, get_dataloader

def setup_experiment():
    """Setup for the experiment: create directories, set seed, etc."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Warning: CUDA is not available, using CPU instead. This will be significantly slower.")
    
    return device

def experiment_resolution_extrapolation(device):
    """Run the resolution extrapolation experiment."""
    print("\n" + "="*80)
    print("Experiment 1: Resolution Extrapolation and Detail Preservation")
    print("="*80)
    
    fit_model = FaFTModel().to(device)
    faft_model = FaFTModel().to(device)  # In a real scenario, these would be different
    
    resolutions = RESOLUTION_EXP_CONFIG["resolutions"]
    ssim_results = evaluate_resolution_extrapolation(fit_model, faft_model, resolutions, RESULTS_DIR, device)
    
    return ssim_results

def experiment_factorized_conditioning(device):
    """Run the factorized conditioning experiment."""
    print("\n" + "="*80)
    print("Experiment 2: Factorized Conditioning with Partial Prompts")
    print("="*80)
    
    model = ConditionedFaFTModel().to(device)
    
    target_resolution = CONDITIONING_EXP_CONFIG["target_resolution"]
    evaluate_factorized_conditioning(model, target_resolution, RESULTS_DIR, device)

def experiment_auxiliary_loss(device):
    """Run the auxiliary loss training experiment."""
    print("\n" + "="*80)
    print("Experiment 3: Training Stability and Efficiency with Factorized Objectives")
    print("="*80)
    
    print("Creating dummy dataset...")
    data, labels = create_dummy_dataset(
        size=STABILITY_EXP_CONFIG["dataset_size"], 
        image_size=STABILITY_EXP_CONFIG["image_size"]
    )
    dummy_loader = get_dataloader(data, labels, batch_size=STABILITY_EXP_CONFIG["batch_size"])
    
    print("Training FaFT (with auxiliary losses)...")
    model_fa = FaFTModel().to(device)
    loss_fa = train_model(
        model_fa, 
        dummy_loader, 
        num_epochs=STABILITY_EXP_CONFIG["num_epochs"], 
        use_aux_loss=True,
        device=device
    )
    
    print("Training FiT (without auxiliary losses)...")
    model_fit = FaFTModel().to(device)
    loss_fit = train_model(
        model_fit, 
        dummy_loader, 
        num_epochs=STABILITY_EXP_CONFIG["num_epochs"], 
        use_aux_loss=False,
        device=device
    )
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_fa, label="FaFT (Aux Loss)")
    ax.plot(loss_fit, label="FiT (No Aux Loss)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Comparison")
    ax.legend()
    plt.tight_layout()
    plot_filename = f"{RESULTS_DIR}/training_loss_comparison_pair1.pdf"
    plt.savefig(plot_filename, format='pdf')
    plt.close(fig)
    
    print(f"Training loss comparison plot saved as: {plot_filename}")
    
    return loss_fa, loss_fit

def run_tests(device):
    """Run quick tests of all experiments."""
    print("\n" + "="*80)
    print("Running quick tests of all experiments")
    print("="*80)
    
    experiment_resolution_extrapolation(device)
    experiment_factorized_conditioning(device)
    experiment_auxiliary_loss(device)
    
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("="*80)

def main():
    """Main function to run all experiments."""
    start_time = time.time()
    
    device = setup_experiment()
    
    required_libs = [
        "torch", "torchvision", "torchmetrics", 
        "numpy", "matplotlib", "opencv-python", 
        "scikit-image", "tensorboard"
    ]
    print("Required libraries for running this experiment:")
    for lib in required_libs:
        print(f" - {lib}")
    
    if RUN_TEST_ONLY:
        run_tests(device)
    else:
        if RUN_RESOLUTION_EXP:
            experiment_resolution_extrapolation(device)
        
        if RUN_CONDITIONING_EXP:
            experiment_factorized_conditioning(device)
        
        if RUN_STABILITY_EXP:
            experiment_auxiliary_loss(device)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")
    
    global STATUS_ENUM
    STATUS_ENUM = "stopped"
    print(f"Status: {STATUS_ENUM}")

if __name__ == "__main__":
    main()
