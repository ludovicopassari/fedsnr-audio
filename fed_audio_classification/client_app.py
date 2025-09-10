import json 
from pathlib import Path
import logging
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

import numpy as np

from models.models import TorchModelOptimized, get_model_info, get_parameters, create_model_with_fixed_seed
from client.client import FlowerClient
from dataset_utils.AudioDS import AudioDS

from device_utils import DEVICE
from snr_utils.snr_processing import calculate_dataset_snr_cnr

from config import FED_DATASET_DIR, FLOWER_CONFIG_FILE, PARTITIONING_METADATA_IID, PARTITIONING_METADATA_DIRICHLET, CLIENT_CONFIG_FILE


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# Hyperparameters
BATCH_SIZE = 32
NUM_CLASSES = 10
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
TARGET_SAMPLE_RATE = 22050
FEATURE_EXT_TYPE = 'mel-spectrogram'

#questi mettili nel file di configurazione fl-config
DISTRIBUTION = 'iid'


with open(CLIENT_CONFIG_FILE) as f:
    client_config = json.load(f)


def load_datasets(partition_id):

    if DISTRIBUTION == 'iid':
        partitioning_metadata_file = PARTITIONING_METADATA_IID
    elif DISTRIBUTION == 'dirichlet':
        partitioning_metadata_file = PARTITIONING_METADATA_DIRICHLET


    try:
        training_data = AudioDS(
            data_path=FED_DATASET_DIR, 
            folds=[1, 2, 3, 4, 5, 6, 7, 8], 
            sample_rate=22050, 
            feature_ext_type='mel-spectrogram', 
            training=True,
            partition_id=partition_id,
            metadata_filename= partitioning_metadata_file
        )

        validation_data = AudioDS(
            data_path=FED_DATASET_DIR, 
            folds=[9], 
            sample_rate=22050, 
            feature_ext_type='mel-spectrogram',
            partition_id=partition_id,
            metadata_filename= partitioning_metadata_file,
            training=False  # Explicit training=False for validation
        )
        
        train_dataloader = DataLoader(
                    training_data, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True,
                    drop_last=True
                )
        
        validation_dataloader = DataLoader(
                        validation_data, 
                        batch_size=BATCH_SIZE, 
                        shuffle=False,  # No need to shuffle validation data
                        drop_last=True
                )
        return  train_dataloader, validation_dataloader
                
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise


def client_fn(context: Context):
    # Usa la stessa funzione di inizializzazione del server
    net = create_model_with_fixed_seed(seed=42).to(DEVICE)  # Stesso seed del server!
    
    partition_id = context.node_config["partition-id"]
    train_loader, validation_loader = load_datasets(partition_id=partition_id)
    
    mri_parameters = calculate_dataset_snr_cnr(train_loader)
    logger.info("ho finito la clinet_fn e sto ritornando Flower client")
    return FlowerClient(
        partition_id=partition_id, 
        net=net, 
        trainloader=train_loader, 
        valloader=validation_loader, 
        config=client_config,
        mean_snr= mri_parameters,
        device= DEVICE
    ).to_client()

# Create the ClientApp
app = ClientApp(client_fn=client_fn)