import numpy as np
from torch.utils.data import DataLoader

def calculate_snr(spectrogram):
    # Calcola la media e la deviazione standard dell'intera immagine
    mean_signal = np.mean(spectrogram)
    std_noise = np.std(spectrogram)

    # Evita la divisione per zero
    if std_noise == 0:
        return float('inf')  # SNR molto alto se non c'è rumore (poco realistico ma utile per evitare errori)
    
    return mean_signal / std_noise

def calculate_dataset_snr_cnr(dataset):
    #print('snr processing')
    total_snr = 0
    num_images = 0
    total_dataset = dataset.dataset  # Se il dataset è un DataLoader, accedi al dataset sottostante
    new_dataloader = DataLoader(total_dataset, batch_size=1)  # Creare un DataLoader per accedere ai singoli elementi
    
    for _, data in enumerate(new_dataloader): 
        image, = data[0]
        image_array = image.numpy()
        
        # Calcolo di SNR e CNR per l'immagine corrente
        snr = calculate_snr(image_array)
    
        total_snr += snr
        num_images += 1
    
    # Calcola la media
    mean_snr = total_snr / num_images

    return mean_snr