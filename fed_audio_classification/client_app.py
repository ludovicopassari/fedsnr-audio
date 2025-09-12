
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import numpy as np

#moduli custom
from device_utils import DEVICE
from config import *
from snr_utils.snr_processing import calculate_dataset_snr_cnr
from logger_config import get_logger

from dataset_utils.AudioDS import AudioDS
from models.models import TorchModelOptimized, get_model_info, get_parameters, create_model_with_fixed_seed
from client.client import FlowerClient

#moduli di sistema
import random
import json 
import os

#ottiene il logger
logger = get_logger(__name__)

def set_all_seeds(seed: int = 42):
    """Set seeds for all random number generators."""
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_datasets(partition_id):

    distribution_type = fl_config['distribution']        

    if distribution_type == 'iid':
        partitioning_metadata_file = PARTITIONING_METADATA_IID
    elif distribution_type == 'dirichlet':
        partitioning_metadata_file = PARTITIONING_METADATA_DIRICHLET

    try:
        training_data = AudioDS(
            data_path=FED_DATASET_DIR, 
            folds=client_config['client_train_folds'], 
            sample_rate=22050,
            training=True,
            partition_id=partition_id,
            metadata_filename= partitioning_metadata_file,
            aug=True,
            num_aug=3,
            aug_prob= 0.3
        )

        """ validation_data = AudioDS(
            data_path=FED_DATASET_DIR, 
            folds=client_config['client_validation_folds'], 
            sample_rate=client_config['sample_rate'], 
            partition_id=partition_id,
            metadata_filename= partitioning_metadata_file,
            training=False,
            aug=False
        )
         """
        logger.info(f"[CLIENT ID : {partition_id}] trains on {len(training_data)} samples")
        #logger.info(f"[CLIENT ID : {partition_id}] validates on {len(validation_data)} samples")
        
        train_dataloader = DataLoader(
                    training_data, 
                    batch_size=client_config['batch_size'], 
                    shuffle=True,
                    drop_last=True,
                )
        
        """ validation_dataloader = DataLoader(
                        validation_data, 
                        batch_size=client_config['batch_size'], 
                        shuffle=False,
                        drop_last=True
                ) """
        return  train_dataloader #, validation_dataloader
                
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]

    net = create_model_with_fixed_seed(seed=42 + partition_id).to(DEVICE)

    torch.manual_seed(42)

    train_loader = load_datasets(partition_id=partition_id)
    
    set_all_seeds(42)
    mri_parameters = calculate_dataset_snr_cnr(train_loader)

    return FlowerClient(
        partition_id=partition_id, 
        net=net, 
        trainloader=train_loader, 
        #valloader=validation_loader, 
        config=client_config,
        mean_snr= mri_parameters,
        device= DEVICE
    ).to_client()


set_all_seeds(seed=42)
app = ClientApp(client_fn=client_fn)