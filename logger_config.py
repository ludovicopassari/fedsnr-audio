import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Restituisce un logger configurato.
    Tutti i moduli possono richiamare questa funzione.
    """
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():  # Evita di aggiungere più handler se già presente
        logger.setLevel(logging.DEBUG)  # Livello globale, modificabile
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)  # Livello del handler
        console_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
           
    return logger
