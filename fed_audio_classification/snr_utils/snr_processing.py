import numpy as np
from torch.utils.data import DataLoader
import torch
from logger_config import get_logger
import pickle
import os
from config import SNR_CACHE_DIR

logger = get_logger(__name__)

def mel_snr_from_spectrogram(S_db, eps_db=5.0):
    """
    Calcola l'SNR stimato dato uno spettrogramma Mel già in dB.
    Input: S_db [B, n_mels, time]
    """
    
    batch, n_mels, T = S_db.shape
    
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


def calculate_dataset_snr(dataset, client_id, eps_db=5.0):
    """
    Calcola uno SNR medio scalare per tutto il dataset.
    Restituisce un float scalare rappresentativo.
    """
    """ SNR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = SNR_CACHE_DIR / f"snr_cache_client_{client_id}.pkl"

    # Controlla se esiste la cache su file
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            normalized_snr = pickle.load(f)
        logger.info(f"[Client {client_id}] SNR letto dalla cache: {normalized_snr:.4f}")
        return normalized_snr """

    total_dataset = getattr(dataset, "dataset", dataset)  # compatibile anche se dataset è DataLoader
    new_dataloader = DataLoader(total_dataset, batch_size=1, shuffle=False)

    snr_list = []

    for _, (spec,lable) in enumerate(new_dataloader):
        
        #spec's shape [batch, channel, mels, time_frame]
        num_batch, num_channel , n_mels, time_frame = spec.shape

        if num_channel > 1:
            image = spec.mean(dim=1)  #[batch, mels, time_frame]
  
        snr = mel_snr_from_spectrogram(image, eps_db=eps_db)  # tensor [B]
        
        snr_list.append(snr)

    # Concatena tutti i tensori e calcola media scalare
    snr_all = torch.cat(snr_list)
    mean_snr = snr_all.mean().item()  # float scalare

    snr_min = snr_all.min().item()
    snr_max = snr_all.max().item()

    normalized_snr = (mean_snr - snr_min) / (snr_max - snr_min)  # tra 0 e 1
    #normalized_snr = max(0.0, min(1.0, normalized_snr))  

    # Salva in cache su file
    """ with open(cache_file, "wb") as f:
        pickle.dump(normalized_snr, f) """

    logger.info(f"[Client {client_id}] SNR normalizzato calcolato")

    return normalized_snr
