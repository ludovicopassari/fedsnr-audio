import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from flwr.common import Metrics, Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from models.models import get_parameters, set_parameters, create_model_with_fixed_seed
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from device_utils import DEVICE
from dataset_utils.AudioDS import AudioDS
from custom_strategies.strategies import FedSNR

from config import *
from logger_config import get_logger

import csv
import os

from fed_audio_classification.snr_utils.snr_processing import calculate_dataset_snr

logger = get_logger(__name__)

def save_metrics_to_csv(round: int, val_loss: float, val_acc: float, precision, recall, f1_score,filename=RESULTS_CSV_FILE):
    # Controlla se il file esiste già
    file_exists = os.path.isfile(filename)
    
    # Scrive i dati nel CSV
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        # Scrive l'intestazione se il file non esiste
        if not file_exists:
            writer.writerow(["round", "val_loss", "accuracy", "precision", "recall", "f1_score"])
        writer.writerow([round, val_loss, val_acc, precision, recall, f1_score])

def get_initial_parameters():
    """Get initial model parameters with consistent initialization"""
    model = create_model_with_fixed_seed(seed=42)
    model.to(DEVICE)
    parameters = get_parameters(model)

    return parameters

def test_with_metrics(model, test_loader, device):
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_true_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            val_predictions.extend(predicted.cpu().numpy())
            val_true_labels.extend(targets.cpu().numpy())
    
    # Calcolo delle metriche usando sklearn
    accuracy = accuracy_score(val_true_labels, val_predictions)
    precision = precision_score(val_true_labels, val_predictions, average='weighted', zero_division=0)
    recall = recall_score(val_true_labels, val_predictions, average='weighted', zero_division=0)
    f1 = f1_score(val_true_labels, val_predictions, average='weighted', zero_division=0)
    avg_val_loss = val_loss / len(test_loader)
    
    # Convertire accuracy in percentuale per compatibilità
    accuracy_percent = accuracy * 100
    
    return {
        'accuracy': accuracy_percent,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': avg_val_loss
    }


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
    """Return a callback that evaluates the global model and saves results to CSV."""
    def evaluate(server_round: int, parameters, config):
        model = create_model_with_fixed_seed(seed=42)
        model.to(device)
        set_parameters(model, parameters)
        #val_acc, avg_val_loss = test(model, testloader, device)
        res_test = test_with_metrics(model=model,  test_loader=testloader, device=device)
        save_metrics_to_csv(server_round, res_test['loss'], res_test['accuracy'], res_test['precision'], res_test['recall'], res_test['f1_score'])
        """
        {
        'accuracy': accuracy_percent,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': avg_val_loss
        }
        """
        return res_test['loss'], {"accuracy": res_test['accuracy']}

    return evaluate

def get_test_dataloader():
    """ Questa funzione restituisce il dataloader relativo al test del modello globale"""

    distribution_type = fl_config['distribution']
    partitioning_metadata_file = ''
    if distribution_type == 'iid':
        partitioning_metadata_file = PARTITIONING_METADATA_IID
    elif distribution_type == 'dirichlet':
        partitioning_metadata_file = PARTITIONING_METADATA_DIRICHLET

    test_data = AudioDS(
            data_path=FED_DATASET_DIR, 
            folds=client_config['server_test_folds'], 
            sample_rate=client_config['sample_rate'],
            partition_id= -1,
            metadata_filename=partitioning_metadata_file,
            training=False,
            aug=False,
            validation=True
        )
    
    return DataLoader(
                    test_data, 
                    batch_size=client_config['batch_size'], 
                    shuffle=False
                )


def on_fit_config(server_round: int) -> dict:
    """FIX: Learning rate che DECRESCE nel tempo"""
    initial_lr = client_config['learning_rate']
    decay_factor = 0.95
    min_lr = 0.0001
    
    # Learning rate che decresce esponenzialmente
    lr = max(initial_lr * (decay_factor ** (server_round - 1)), min_lr)
    
    logger.info(f"Round {server_round}: Using learning rate {lr:.6f}")
    return {"lr": lr}


def aggregate_fit_metrics(metrics):
    # metrics è una lista di tuple: (num_examples, metrics_dict)
    # esempio: [(2000, {"accuracy": 0.8, "mean_snr": 15.2}), (1500, {"accuracy": 0.75, "mean_snr": 12.7})]

    total_examples = sum(num_examples for num_examples, _ in metrics)

    # Accuracy aggregata pesata
    #acc = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples

    # SNR aggregata pesata
    snr = sum(num_examples * m["mean_snr"] for num_examples, m in metrics) / total_examples

    return {"mean_snr": snr}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Get initial parameters
    initial_parameters = ndarrays_to_parameters(get_initial_parameters())
    
    #dataloader relativo ai dati di test del modello globale
    test_dataloader = get_test_dataloader()
    
    
    config = ServerConfig(num_rounds=fl_config['num_rounds'])

    # Create FedAvg strategy 
    strategyAVG = FedAvg(
        fraction_fit= fl_config['fraction_fit'],
        fraction_evaluate=fl_config['fraction_evaluate'],  
        min_fit_clients=fl_config['min_fit_clients'],
        min_evaluate_clients=fl_config['min_evaluate_clients'],
        min_available_clients=fl_config['min_available_clients'],
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(test_dataloader, DEVICE),
        on_fit_config_fn=on_fit_config,
    )

    #Create FedAvg strategy with initial parameters
    strategySNR = FedSNR(
        fraction_fit= fl_config['fraction_fit'],
        fraction_evaluate=fl_config['fraction_evaluate'],  
        min_fit_clients=fl_config['min_fit_clients'],
        min_evaluate_clients=fl_config['min_evaluate_clients'],
        min_available_clients=fl_config['min_available_clients'],
        on_fit_config_fn=on_fit_config,
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(test_dataloader, device=DEVICE),
    )

    sel_strategie = fl_config["strategy"]
    if sel_strategie == "FedAvg":
        strategy = strategyAVG
    elif sel_strategie == "FedSNR":
        strategy = strategySNR
    else:
        raise ValueError(f"Invalid strategy: {fl_config['strategy']}")
    
    logger.info(f"SERVER -> strategie used {sel_strategie} ")
    return ServerAppComponents(strategy=strategy, config=config)

# Crea la ServerApp
app = ServerApp(server_fn=server_fn)