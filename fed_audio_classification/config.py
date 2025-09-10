from pathlib import Path

# Root del progetto
BASE_DIR = Path(__file__).resolve().parent.parent
#/Users/ludovicopassari/Documents/university/tesi/project/fedsnr

# Dataset
FED_DATASET_DIR = BASE_DIR.parent / "fed_dataset"

PARTITIONING_METADATA_IID = FED_DATASET_DIR / "partitioning-metadata" / "iid-partitioning-metadata.csv"
PARTITIONING_METADATA_DIRICHLET = FED_DATASET_DIR / "partitioning-metadata" / "dirichlet-partitioning-metadata.csv"

# Client config
CLIENT_CONFIG_FILE = BASE_DIR / "fed_audio_classification" / "client" / "client_config.json"
FLOWER_CONFIG_FILE = BASE_DIR / "fed_audio_classification" / "fl_config.json"

print(BASE_DIR)