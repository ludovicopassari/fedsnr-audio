
import random
import os
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_global_seed(seed: int = 42) -> None:
    """
    Set seeds for all random number generators to ensure reproducibility.
    
    Args:
        seed (int): The seed value to use for all random number generators
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA settings for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility - may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Global seed set to {seed}")

def get_client_seed(base_seed: int, client_id: int) -> int:
    """
    Generate a unique seed for each client based on a base seed and client ID.
    
    Args:
        base_seed (int): The base seed value
        client_id (int): The client identifier
        
    Returns:
        int: Unique seed for the client
    """
    return base_seed + client_id

def set_client_seed(base_seed: int, client_id: int) -> None:
    """
    Set seeds for a specific client to ensure reproducibility while maintaining
    uniqueness between clients.
    
    Args:
        base_seed (int): The base seed value
        client_id (int): The client identifier
    """
    client_seed = get_client_seed(base_seed, client_id)
    set_global_seed(client_seed)
    logger.info(f"Client {client_id} seed set to {client_seed}")

def ensure_deterministic_operations() -> None:
    """
    Ensure PyTorch operations are deterministic for reproducibility.
    Note: This may impact performance.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("Deterministic operations enabled")

def seed_worker(worker_id: int) -> None:
    """
    Seed worker function for DataLoader to ensure reproducibility in multi-process data loading.
    
    Args:
        worker_id (int): The worker process ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_generator(seed: int) -> torch.Generator:
    """
    Create a torch.Generator with the specified seed for DataLoader.
    
    Args:
        seed (int): The seed value
        
    Returns:
        torch.Generator: Seeded generator
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator