from config import *
from snr_utils.snr_processing import calculate_dataset_snr_cnr  # make sure this exists
from torch.utils.data import DataLoader
from dataset_utils.AudioDS import AudioDS

# Main loop
for i in range(10):
    train_data_partition = AudioDS(
        data_path=FED_DATASET_DIR, 
        folds=client_config['client_train_folds'], 
        sample_rate=22050,
        training=True,
        partition_id=i,
        metadata_filename=PARTITIONING_METADATA_IID,
        aug=False
    )
    
    train_dataloader = DataLoader(
        train_data_partition, 
        batch_size=client_config['batch_size'], 
        shuffle=True,
        drop_last=True,
    )
    
    mean_snr_partition = calculate_dataset_snr_cnr(train_dataloader)
    print(f"Partition {i} mean SNR: {mean_snr_partition:.4f}")
