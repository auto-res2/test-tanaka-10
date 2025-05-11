"""
Data preprocessing for the FaFT experiment.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset

def create_dummy_dataset(size=50, image_size=64, channels=3):
    """Create a dummy dataset for testing."""
    data = torch.randn(size, channels, image_size, image_size)
    labels = torch.zeros(size)  # Dummy labels
    return data, labels

def get_dataloader(data, labels, batch_size=5, shuffle=True):
    """Create a DataLoader from data and labels."""
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def compute_edge_ssim(img1, img2):
    """Compute SSIM on edge maps extracted via the Canny operator."""
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img1_gray = cv2.cvtColor((img1_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor((img2_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges1 = cv2.Canny(img1_gray, 100, 200)
    edges2 = cv2.Canny(img2_gray, 100, 200)
    from skimage.metrics import structural_similarity as compare_ssim
    ssim_edge, _ = compare_ssim(edges1, edges2, full=True)
    return ssim_edge
