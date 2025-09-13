from config import *
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_utils.AudioDS import AudioDS
from typing import Union, Tuple, Optional

def mel_snr_from_spectrogram(S_db, eps_db=5.0):
    """
    Calcola l'SNR stimato dato uno spettrogramma Mel già in dB.
    Input: S_db [B, n_mels, time]
    """
    B, channel, n_mels, T = S_db.shape
    
    # Rumore di fondo per sample
    noise_floor = S_db.flatten(1).median(dim=1).values  # [B]
    
    # Maschera per valori sopra soglia
    mask = S_db > (noise_floor[:, None, None] + eps_db)
    
    # Energia del segnale sopra soglia
    signal_energy = torch.where(mask, S_db, torch.tensor(float('nan'), device=S_db.device))
    signal_energy = torch.nanmean(signal_energy, dim=(1, 2))  # [B]
    
    # Sostituisci eventuali NaN con noise_floor
    signal_energy = torch.where(torch.isnan(signal_energy), noise_floor, signal_energy)
    
    snr_db = signal_energy - noise_floor
    return snr_db


def calculate_dataset_snr_cnr(dataset):
    total_dataset = getattr(dataset, "dataset", dataset)  # compatibile anche se dataset è DataLoader
    new_dataloader = DataLoader(total_dataset, batch_size=1, shuffle=False)
    snr_list = []

    for data in new_dataloader:
        
        spectrogram = data[0]  #[batch, channel , n_mels, time]
    
        snr = mel_snr_from_spectrogram(spectrogram)  # tensor [1]
        snr_list.append(snr)

    # concatena tutti i tensori
    snr_all = torch.cat(snr_list)  # [num_samples]
    mean_snr = snr_all.mean().item()  # float scalare

    logger.info(f"Mean SNR {mean_snr:.2f} dB")
    return mean_snr


for i in range(10):

    
    train_data_partition = AudioDS(
        data_path=FED_DATASET_DIR, 
        folds=client_config['client_train_folds'], 
        sample_rate=22050,
        training=True,
        partition_id=i,
        metadata_filename=PARTITIONING_METADATA_IID,
        aug=False  # Importante: no augmentation per calcolo SNR consistente
    )
    
    train_dataloader = DataLoader(
        train_data_partition, 
        batch_size=client_config['batch_size'], 
        shuffle=False,  # No shuffle per calcolo deterministico
        drop_last=True,
    )
    
    # Calcola SNR per questa partizione
    mean_snr_partition = calculate_dataset_snr_cnr(
        train_dataloader
    )
    
    print(f"Partition {i} - mean snr {mean_snr_partition}")

