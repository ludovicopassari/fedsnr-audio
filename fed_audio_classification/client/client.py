from flwr.client import NumPyClient, ClientApp
from models.models import get_parameters, set_parameters


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_utils.AudioDS import AudioDS
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import os


import random
from logger_config import get_logger


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


logger = get_logger(__name__)


class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader,mean_snr,config, device):
        set_all_seeds(seed=42 + partition_id)

        #model settings
        self._net = net
        self._epoch = config['epochs'] if "epochs" in config else 30
        self.batch_size = config['batch_size'] if 'batch_size' in config else 32
        self.learning_rate = config['learning_rate'] if 'learning_rate' in config else 0.001
        self.weight_decay = config['weight_decay'] if 'weight_decay' in config else 1e-4
        self._device = device
        self.criterion= None
        self.optimizer = None

        #data partition setting
        self.partition_id = partition_id
        self._trainloader = trainloader
        

        #aggregation
        self.mean_snr = float(mean_snr)
        
    def get_parameters(self, config):
        return get_parameters(self._net)

    def fit(self, parameters, config):
        set_parameters(self._net, parameters)
        
        if "lr" in config:
            self.learning_rate = config["lr"]
    
        self.train()

        metrics = {
            "client_id": self.partition_id,
            "mean_snr": self.mean_snr
        }

        return get_parameters(self._net), len(self._trainloader), metrics

    """ def evaluate(self, parameters, config):
        set_parameters(self._net, parameters)
        criterion = nn.CrossEntropyLoss()
        val_acc, val_loss, val_precision, val_recall, val_f1 = self.validate_epoch(criterion=criterion)
        return float(val_loss), len(self._validation_loader), {"accuracy": float(val_acc)} """
    
    def calculate_metrics(self, predictions, true_labels):
        """Calculate precision, recall, and F1 score"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        return precision, recall, f1
    
    def validate_epoch(self, criterion):
        self._net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for data, targets in self._validation_loader:
                data, targets = data.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                outputs = self._net(data)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, targets)

                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(targets.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(self._validation_loader)
        val_precision, val_recall, val_f1 = self.calculate_metrics(val_predictions, val_true_labels)
        
        return val_acc, avg_val_loss, val_precision, val_recall, val_f1
        
    def train_epoch(self,criterion, optimizer):
        self._net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_predictions = []
        train_true_labels = []
        
        for batch_idx, (data, targets) in enumerate(self._trainloader):
            data, targets = data.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = self._net(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            #torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            train_predictions.extend(predicted.cpu().numpy())
            train_true_labels.extend(targets.cpu().numpy())
            
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(self._trainloader)
        train_precision, train_recall, train_f1 = self.calculate_metrics(train_predictions, train_true_labels)
        
        return train_acc, avg_train_loss, train_precision, train_recall, train_f1
    
    def train(self):

        self._net.to(self._device)

        logger.info(f"Client {self.partition_id} using lr={self.learning_rate}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._net.parameters(), lr= self.learning_rate, weight_decay=self.weight_decay)

        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        metrics_tracker = MetricsTracker()
        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 10
        
        logger.info(f"Starting training for {self._epoch} epochs...")
        
        for epoch in range(self._epoch):
            
            train_metrics = self.train_epoch(criterion, optimizer)
            #val_metrics = self.validate_epoch(criterion=criterion)
            
            """ metrics_tracker.update(train_metrics, val_metrics)
            
            train_acc, train_loss, train_precision, train_recall, train_f1 = train_metrics
            val_acc, val_loss, val_precision, val_recall, val_f1 = val_metrics

            logger.info(
                f"[Client {self.partition_id}] [{epoch +1}/{self._epoch}]"
                f"Train -> Acc: {train_acc:.4f}, Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, "
                f"Recall: {train_recall:.4f}, F1: {train_f1:.4f} | "
                f"Val -> Acc: {val_acc:.4f}, Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, "
                f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
            ) """
                        
        return 

    
class MetricsTracker:
    def __init__(self):
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_precision_history = []
        self.train_recall_history = []
        self.train_f1_history = []
        self.val_precision_history = []
        self.val_recall_history = []
        self.val_f1_history = []
    
    def update(self, train_metrics, val_metrics):
        """Update all metrics"""
        train_acc, train_loss, train_precision, train_recall, train_f1 = train_metrics
        val_acc, val_loss, val_precision, val_recall, val_f1 = val_metrics
        
        self.train_acc_history.append(train_acc)
        self.train_loss_history.append(train_loss)
        self.train_precision_history.append(train_precision)
        self.train_recall_history.append(train_recall)
        self.train_f1_history.append(train_f1)
        
        self.val_acc_history.append(val_acc)
        self.val_loss_history.append(val_loss)
        self.val_precision_history.append(val_precision)
        self.val_recall_history.append(val_recall)
        self.val_f1_history.append(val_f1)


