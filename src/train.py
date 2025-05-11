"""
Model implementation and training code for the FaFT experiment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image, make_grid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.faft_config import *


class FrequencyBranch(nn.Module):
    def __init__(self, band):
        super(FrequencyBranch, self).__init__()
        self.band = band
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.net(x)

class FaFTModel(nn.Module):
    def __init__(self):
        super(FaFTModel, self).__init__()
        self.low_freq = FrequencyBranch("low")
        self.mid_freq = FrequencyBranch("mid")
        self.high_freq = FrequencyBranch("high")
    
    def forward(self, x):
        out_low = self.low_freq(x)
        out_mid = self.mid_freq(x)
        out_high = self.high_freq(x)
        fused = (out_low + out_mid + out_high) / 3.0
        return fused
        
    def get_branch_outputs(self, x):
        """Get individual branch outputs for auxiliary loss computation."""
        out_low = self.low_freq(x)
        out_mid = self.mid_freq(x)
        out_high = self.high_freq(x)
        return out_low, out_mid, out_high

class ConditionedFrequencyBranch(nn.Module):
    def __init__(self, band):
        super(ConditionedFrequencyBranch, self).__init__()
        self.band = band
        self.condition_on_prompt = False  # This flag controls conditioning
        self.condition_proj = nn.Linear(512, 3)  # Changed from 16 to 3 to match input channels
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x, prompt_embedding=None):
        if self.condition_on_prompt and prompt_embedding is not None:
            cond = self.condition_proj(prompt_embedding).unsqueeze(-1).unsqueeze(-1)
            cond = cond.expand(-1, -1, x.size(2), x.size(3))
            x = x + cond
        return self.net(x)

class ConditionedFaFTModel(nn.Module):
    def __init__(self):
        super(ConditionedFaFTModel, self).__init__()
        self.low_freq = ConditionedFrequencyBranch("low")
        self.mid_freq = ConditionedFrequencyBranch("mid")
        self.high_freq = ConditionedFrequencyBranch("high")
    
    def forward(self, x, prompt_embedding=None, band_conditioning={'low': False, 'mid': False, 'high': False}):
        self.low_freq.condition_on_prompt = band_conditioning.get('low', False)
        self.mid_freq.condition_on_prompt = band_conditioning.get('mid', False)
        self.high_freq.condition_on_prompt = band_conditioning.get('high', False)
        
        out_low = self.low_freq(x, prompt_embedding)
        out_mid = self.mid_freq(x, prompt_embedding)
        out_high = self.high_freq(x, prompt_embedding)
        out = (out_low + out_mid + out_high) / 3.0
        return out


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
    
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)

def train_model(model, dataloader, num_epochs=2, use_aux_loss=True, device="cuda"):
    """Training function that compares auxiliary losses vs. a single global objective."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    l1_loss_fn = nn.L1Loss()
    perceptual_loss_fn = PerceptualLoss()
    loss_log = []
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            main_loss = l1_loss_fn(outputs, images)  # Global reconstruction loss
            
            if use_aux_loss:
                if isinstance(model, FaFTModel):
                    low_out, mid_out, high_out = model.get_branch_outputs(images)
                    aux_loss_low = l1_loss_fn(low_out, images)         # low-frequency L1 loss
                    aux_loss_mid = l1_loss_fn(mid_out, images)         # mid-frequency L1 loss
                    aux_loss_high = perceptual_loss_fn(high_out, images)  # high-frequency perceptual loss
                    total_loss = main_loss + 0.5 * aux_loss_low + 0.3 * aux_loss_mid + 0.5 * aux_loss_high
                else:
                    low_out = model.low_freq(images)
                    high_out = model.high_freq(images)
                    aux_loss_low = l1_loss_fn(low_out, images)         # low-frequency L1 loss
                    aux_loss_high = perceptual_loss_fn(high_out, images)  # high-frequency perceptual loss
                    total_loss = main_loss + 0.5 * aux_loss_low + 0.5 * aux_loss_high
            else:
                total_loss = main_loss
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            loss_log.append(total_loss.item())
            
            if i % 5 == 0:
                print(f"Epoch {epoch+1}, Iteration {i}, Loss: {total_loss.item():.4f}")
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return loss_log
