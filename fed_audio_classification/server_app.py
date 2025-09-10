from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import json
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from models.models import TorchModelOptimized, get_parameters, set_parameters,create_model_with_fixed_seed
from device_utils import DEVICE
from dataset_utils.AudioDS import AudioDS
from custom_strategies.strategies import FedSNR

from config import FED_DATASET_DIR, FLOWER_CONFIG_FILE, PARTITIONING_METADATA_IID, PARTITIONING_METADATA_DIRICHLET

import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


with open(FLOWER_CONFIG_FILE) as f:
    fl_config = json.load(f)

def get_initial_parameters():
    """Get initial model parameters with consistent initialization"""
    model = create_model_with_fixed_seed(seed=42)  # Fixed seed for reproducibility
    model.to(DEVICE)
    parameters = get_parameters(model)

    return parameters

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    print({"accuracy": sum(accuracies) / sum(examples)})

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def test(model, test_loader, device):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(data)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, targets)

                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(targets.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)
        
        return val_acc, avg_val_loss

def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model."""
    def evaluate(server_round: int, parameters, config):
        model = create_model_with_fixed_seed(seed=42)
        model.to(device)
        set_parameters(model, parameters)
        val_acc, avg_val_loss = test(model, testloader, DEVICE)
        
        return avg_val_loss, {"accuracy": val_acc}

    return evaluate

def get_test_dataloader():
    distribution_type = fl_config['distribution']
    
    partitioning_metadata_file = ''
    if distribution_type == 'iid':
        partitioning_metadata_file = PARTITIONING_METADATA_IID
    elif distribution_type == 'dirichlet':
        partitioning_metadata_file = PARTITIONING_METADATA_DIRICHLET

    test_data = AudioDS(
            data_path=FED_DATASET_DIR, 
            folds=[10], 
            sample_rate=22050, 
            feature_ext_type='mel-spectrogram',
            partition_id= -1,
            metadata_filename= partitioning_metadata_file,
            training=False 
        )
    

    return DataLoader(
                    test_data, 
                    batch_size=32, 
                    shuffle=False
                )


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    logger.info("start server_fn")
    # Get initial parameters
    initial_parameters = ndarrays_to_parameters(get_initial_parameters())
    

    test_dataloader = get_test_dataloader()


    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=fl_config['num_rounds'])

    # Create FedAvg strategy with initial parameters
    strategyAVG = FedAvg(
        fraction_fit= fl_config['fraction_fit'],
        fraction_evaluate=fl_config['fraction_evaluate'],  
        min_fit_clients=fl_config['min_fit_clients'],
        min_evaluate_clients=fl_config['min_evaluate_clients'],
        min_available_clients=fl_config['min_available_clients'],
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn= weighted_average,
        evaluate_fn=get_evaluate_fn(test_dataloader, DEVICE)
        
    )

    #Create FedAvg strategy with initial parameters
    strategySNR = FedSNR(
        fraction_fit=fl_config["fraction_fit"],
        fraction_evaluate=fl_config['fraction_evaluate'],
        min_available_clients=fl_config["fitClients"],
        #on_fit_config_fn=on_fit_config,
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(test_dataloader, device=DEVICE),
    )

    if fl_config["strategy"] == "FedAvg":
        strategy = strategyAVG
    elif fl_config["strategy"] == "FedSNR":
        strategy = strategySNR
    else:
        raise ValueError(f"Invalid strategy: {fl_config['strategy']}")

    return ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp
app = ServerApp(server_fn=server_fn)