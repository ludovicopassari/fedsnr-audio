from pathlib import Path
import json
import logging

# Configurazione logger di base
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Root del progetto
BASE_DIR = Path(__file__).resolve().parent.parent
# /Users/ludovicopassari/Documents/university/tesi/project/fedsnr

# Dataset
FED_DATASET_DIR = BASE_DIR.parent / "augmented_dataset"

PARTITIONING_METADATA_IID = FED_DATASET_DIR / "partitioning-metadata" / "iid-partitioning-metadata.csv"
PARTITIONING_METADATA_DIRICHLET = FED_DATASET_DIR / "partitioning-metadata" / "dirichlet-partitioning-metadata.csv"

# Client config
CLIENT_CONFIG_FILE = BASE_DIR / "fed_audio_classification" / "client" / "client_config.json"
FLOWER_CONFIG_FILE = BASE_DIR / "fed_audio_classification" / "fl_config.json"

# Result file
RESULTS_CSV_FILE = FED_DATASET_DIR / "global_model_results.csv"


def open_config_files():
    client_config = None
    fl_config = None

    # Apro e carico il file client_config.json
    try:
        with open(CLIENT_CONFIG_FILE, 'r') as f:
            client_config = json.load(f)
        logger.debug(f"File {CLIENT_CONFIG_FILE} caricato con successo")
    except FileNotFoundError:
        logger.error(f"File non trovato: {CLIENT_CONFIG_FILE}")
    except json.JSONDecodeError as e:
        logger.error(f"Errore nel parsing JSON di {CLIENT_CONFIG_FILE}: {e}")
    except Exception as e:
        logger.exception(f"Errore imprevisto durante l'apertura di {CLIENT_CONFIG_FILE}: {e}")

    # Apro e carico il file fl_config.json
    try:
        with open(FLOWER_CONFIG_FILE, 'r') as f:
            fl_config = json.load(f)
        logger.debug(f"File {FLOWER_CONFIG_FILE} caricato con successo")
    except FileNotFoundError:
        logger.error(f"File non trovato: {FLOWER_CONFIG_FILE}")
    except json.JSONDecodeError as e:
        logger.error(f"Errore nel parsing JSON di {FLOWER_CONFIG_FILE}: {e}")
    except Exception as e:
        logger.exception(f"Errore imprevisto durante l'apertura di {FLOWER_CONFIG_FILE}: {e}")

    return client_config, fl_config


client_config, fl_config = open_config_files()