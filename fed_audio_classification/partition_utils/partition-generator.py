from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np

from config import BASE_DIR, FED_DATASET_DIR

#dataset info
dataset_dir = FED_DATASET_DIR
dataset_metadata = dataset_dir / 'metadata' / 'UrbanSound8K_augmented.csv'

config_file = BASE_DIR / 'fed_audio_classification' / 'partition_utils' / 'partitioner_config.json' # Percorso del file JSON

config = None
with open(config_file, "r") as f:
    config = json.load(f)

# Accesso ai valori
NUM_PARTITIONS = config["num_partition"]
PARTITION_TYPE = config["partition_type"]
TRAINING_FOLDS = config["training_folds"]
START_PARTITION_COUNT = config["start_partition_count"]

urban8k_classes = [
    "air_conditioner",  # 0
    "car_horn",         # 1
    "children_playing", # 2
    "dog_bark",         # 3
    "drilling",         # 4
    "engine_idling",    # 5
    "gun_shot",         # 6
    "jackhammer",       # 7
    "siren",            # 8
    "street_music"      # 9
]

def iid_partitioning(metadata_df):

    iid_partitioning_metadata = metadata_df.copy(deep=True)
    iid_partitioning_metadata['partition_id'] = -1  # inizializzo

    # Dizionario classe -> lista di indici
    class_indices = {
        class_name: metadata_df.index[
            (metadata_df["class"] == class_name) & (metadata_df["fold"].isin(TRAINING_FOLDS))
        ].tolist() 
        for class_name in urban8k_classes
    }
    
    # Dizionario client -> lista di indici
    client_data = {i + START_PARTITION_COUNT: [] for i in range(NUM_PARTITIONS)}
    
    # Dividi gli indici delle classi in modo uniforme tra i client
    for class_label, indices in class_indices.items():
        shuffled_indices = indices.copy()
        np.random.shuffle(shuffled_indices)
        split_indices = np.array_split(shuffled_indices, NUM_PARTITIONS)
        
        for client_idx, client_indices in enumerate(split_indices):
            client_id = client_idx + START_PARTITION_COUNT
            client_data[client_id].extend(client_indices.tolist())
    
    # Mescola gli indici per ogni client
    for client_id in client_data:
        np.random.shuffle(client_data[client_id])
    
    # Aggiorna il DataFrame con l'ID del client
    for client_id, indices in client_data.items():
        iid_partitioning_metadata.loc[indices, "partition_id"] = client_id

    # Stampa distribuzione classi per client (opzionale)
    for client_id, indices in client_data.items():
        subset = metadata_df.loc[indices]
        print(f"Client {client_id}")
        print(subset["class"].value_counts(normalize=True))

    # Salvataggio su CSV
    output_file = Path(dataset_dir / "partitioning-metadata/iid-partitioning-metadata.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    iid_partitioning_metadata.to_csv(output_file, index=True)

    return iid_partitioning_metadata

def dirichlet_partitioning(metadata_df, alpha: float = 0.5):

    dirichlet_partitioning_metadata = metadata_df.copy(deep=True)
    dirichlet_partitioning_metadata['partition_id'] = -1  # inizializzo

    class_indices = {
        class_name: metadata_df.index[
            (metadata_df["class"] == class_name) & (metadata_df["fold"].isin(TRAINING_FOLDS))
        ].tolist() 
        for class_name in urban8k_classes
    }
    
    client_data = {i + START_PARTITION_COUNT: [] for i in range(NUM_PARTITIONS)}
    
    for class_label, indices in class_indices.items():
        proportions = np.random.dirichlet([alpha] * NUM_PARTITIONS)
        shuffled_indices = indices.copy()
        np.random.shuffle(shuffled_indices)
        samples_per_client = (proportions * len(indices)).astype(int)
        
        diff = len(indices) - samples_per_client.sum()
        if diff > 0:
            samples_per_client[:diff] += 1
        elif diff < 0:
            for i in range(-diff):
                max_client = np.argmax(samples_per_client)
                if samples_per_client[max_client] > 0:
                    samples_per_client[max_client] -= 1
        
        current_idx = 0
        for client_idx, num_samples in enumerate(samples_per_client):
            client_id = client_idx + START_PARTITION_COUNT
            if num_samples > 0:
                client_indices = shuffled_indices[current_idx:current_idx + num_samples]
                client_data[client_id].extend(client_indices)
                current_idx += num_samples

    for client_id in client_data:
        np.random.shuffle(client_data[client_id])
    
    for client_id, indices in client_data.items():
        dirichlet_partitioning_metadata.loc[indices, "partition_id"] = client_id

    for client_id, indices in client_data.items():
        subset = metadata_df.loc[indices]
        print(f"Client {client_id}")
        print(subset["class"].value_counts(normalize=True))
    
    output_file = Path(dataset_dir / "partitioning-metadata/dirichlet-partitioning-metadata.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dirichlet_partitioning_metadata.to_csv(output_file, index=True)
    
    return dirichlet_partitioning_metadata        

def main():
    metadata_df = pd.read_csv(dataset_metadata)
    metadata_df.drop(axis='columns', columns=['fsID','start','end','salience'], inplace=True)

    if PARTITION_TYPE == "iid":
        df_partitioned = iid_partitioning(metadata_df)
    elif PARTITION_TYPE == "dirichlet":
        df_partitioned = dirichlet_partitioning(metadata_df, alpha=0.5)

if __name__ == "__main__":
    main()