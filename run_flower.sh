#!/bin/bash

# Nome ambiente
CONDA_ENV="ml_audio"

# Attiva l'ambiente
echo "Attivo ambiente '$CONDA_ENV'..."
conda activate "$CONDA_ENV"

# Setta PYTHONPATH se non già settato
if [ -z "$PYTHONPATH" ]; then
    echo "Imposto PYTHONPATH..."
    export PYTHONPATH="$(pwd)/fed_audio_classification:$PYTHONPATH"
fi

# Avvia Flower
echo "Avvio l'app Flower..."
flwr run .
