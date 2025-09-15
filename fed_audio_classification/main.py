
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from flwr.client import ClientApp
from flwr.server import ServerApp 
from flwr.common import Context
from flwr.simulation import run_simulation

from client_app import client_fn
from server_app import server_fn
import numpy as np

#moduli custom
from device_utils import DEVICE
from config import *
from snr_utils.snr_processing import calculate_dataset_snr
from logger_config import get_logger

from dataset_utils.AudioDS import AudioDS
from models.models import TorchModelOptimized, get_model_info, get_parameters, create_model_with_fixed_seed
from client.client import FlowerClient

#moduli di sistema
import random
import json 
import os
import multiprocessing

# Construct the ClientApp passing the client generation function
client_app = ClientApp(client_fn=client_fn)

# Create your ServerApp passing the server generation function
server_app = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=20,
    backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 0.0}}
)