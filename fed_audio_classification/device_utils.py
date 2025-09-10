import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device(prefer_gpu=True, prefer_mps=True):
    """
    Determina il device da usare per PyTorch.

    Args:
        prefer_gpu (bool): usa CUDA se disponibile
        prefer_mps (bool): usa MPS se disponibile (solo macOS)
    
    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

# Device globale
DEVICE = get_device()
