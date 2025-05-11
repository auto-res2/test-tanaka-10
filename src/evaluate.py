"""
Evaluation code for the FaFT experiment.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.faft_config import *
from src.preprocess import compute_edge_ssim

def resolution_adaptive_sampling(model, target_resolution, device="cuda"):
    """Sample from the model at a target resolution."""
    model = model.to(device)
    input_tensor = torch.randn(1, 3, target_resolution[0] // 4, target_resolution[1] // 4, device=device)
    input_tensor = F.interpolate(input_tensor, size=target_resolution, mode='bilinear', align_corners=False)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    return output

def generate_conditioned_image(model, target_resolution, prompt_embedding, band_cond, device="cuda"):
    """Generate an image with specific conditioning on frequency bands."""
    model = model.to(device)
    input_tensor = torch.randn(1, 3, target_resolution[0] // 4, target_resolution[1] // 4, device=device)
    input_tensor = F.interpolate(input_tensor, size=target_resolution, mode='bilinear', align_corners=False)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor, prompt_embedding, band_conditioning=band_cond)
    
    return output

def evaluate_resolution_extrapolation(fit_model, faft_model, resolutions, results_dir="results", device="cuda"):
    """Evaluate the resolution extrapolation experiment."""
    print("Starting Resolution Extrapolation and Detail Preservation Evaluation")
    
    os.makedirs(results_dir, exist_ok=True)
    
    ssim_results = {}
    
    for res in resolutions:
        im_fit = resolution_adaptive_sampling(fit_model, res, device)
        im_faft = resolution_adaptive_sampling(faft_model, res, device)
        
        ssim_edge = compute_edge_ssim(im_fit, im_faft)
        ssim_results[res] = ssim_edge
        print(f"Resolution {res}: Edge SSIM between FiT and FaFT = {ssim_edge:.4f}")
        
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        im_fit_np = im_fit.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        im_faft_np = im_faft.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        axs[0].imshow(np.clip(im_fit_np, 0, 1))
        axs[0].set_title(f"FiT {res[0]}x{res[1]}")
        axs[0].axis('off')
        axs[1].imshow(np.clip(im_faft_np, 0, 1))
        axs[1].set_title(f"FaFT {res[0]}x{res[1]}")
        axs[1].axis('off')
        plt.tight_layout()
        plot_filename = f"{results_dir}/resolution_extrapolation_{res[0]}x{res[1]}.pdf"
        plt.savefig(plot_filename, format='pdf')
        plt.close(fig)
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    resolutions_str = [f"{w}x{h}" for (w, h) in resolutions]
    ssim_values = [ssim_results[res] for res in resolutions]
    ax2.bar(resolutions_str, ssim_values, color='skyblue')
    ax2.set_xlabel("Resolution")
    ax2.set_ylabel("Edge SSIM")
    ax2.set_title("Edge SSIM between FiT and FaFT across Resolutions")
    plt.tight_layout()
    summary_filename = f"{results_dir}/resolution_extrapolation_summary.pdf"
    plt.savefig(summary_filename, format='pdf')
    plt.close(fig2)
    
    print(f"Resolution extrapolation evaluation completed. Summary plot saved as: {summary_filename}")
    
    return ssim_results

def evaluate_factorized_conditioning(model, target_resolution, results_dir="results", device="cuda"):
    """Evaluate the factorized conditioning experiment."""
    print("Starting Factorized Conditioning with Partial Prompts Evaluation")
    
    os.makedirs(results_dir, exist_ok=True)
    
    prompt_embedding = torch.randn(1, 512, device=device)
    
    band_conditioning_high = {'low': False, 'mid': False, 'high': True}
    band_conditioning_all = {'low': True, 'mid': True, 'high': True}
    
    im_high_cond = generate_conditioned_image(model, target_resolution, prompt_embedding, band_conditioning_high, device)
    im_all_cond = generate_conditioned_image(model, target_resolution, prompt_embedding, band_conditioning_all, device)
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    im_high_np = im_high_cond.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    im_all_np = im_all_cond.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    axs[0].imshow(np.clip(im_high_np, 0, 1))
    axs[0].set_title("High-Frequency Conditioning Only")
    axs[0].axis('off')
    axs[1].imshow(np.clip(im_all_np, 0, 1))
    axs[1].set_title("Full Conditioning (All Branches)")
    axs[1].axis('off')
    plt.tight_layout()
    plot_filename = f"{results_dir}/factorized_conditioning_pair1.pdf"
    plt.savefig(plot_filename, format='pdf')
    plt.close(fig)
    
    print(f"Factorized conditioning evaluation completed. Comparison plot saved as: {plot_filename}")
