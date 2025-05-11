"""
Configuration for the FaFT experiment.
"""

SEED = 42
DEVICE = "cuda"  # Use CUDA for NVIDIA Tesla T4 GPU
RESULTS_DIR = "results"
LOG_DIR = "logs"

MODEL_CONFIG = {
    "base_channels": 16,
    "use_auxiliary_loss": True,
}

TRAIN_CONFIG = {
    "batch_size": 5,
    "num_epochs": 2,
    "learning_rate": 1e-4,
}

RESOLUTION_EXP_CONFIG = {
    "resolutions": [(128, 128), (256, 128), (128, 256), (512, 512)],
}

CONDITIONING_EXP_CONFIG = {
    "target_resolution": (256, 256),
}

STABILITY_EXP_CONFIG = {
    "num_epochs": 2,
    "batch_size": 5,
    "dataset_size": 50,
    "image_size": 64,
}

RUN_RESOLUTION_EXP = True
RUN_CONDITIONING_EXP = True 
RUN_STABILITY_EXP = True
RUN_TEST_ONLY = False  # Set to True for short test runs

STATUS_ENUM = "running"  # Will be set to "stopped" when complete
