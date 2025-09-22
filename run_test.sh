# Setta PYTHONPATH se non già settato
if [ -z "$PYTHONPATH" ]; then
    echo "Imposto PYTHONPATH..."
    export PYTHONPATH="$(pwd)/fed_audio_classification:$PYTHONPATH"
fi


for fitFraction in 0.1 0.3 0.5 0.7; do
    for strategy in FedAvg FedSNR FedSNRCS ; do
        for distribution in iid dirichlet; do
            for percentages in 0.0 0.2 0.5; do
                
                    python3 script_tuning.py --input fed_audio_classification/fl_config.json --output fed_audio_classification/fl_config.json \
                    --fitFraction $fitFraction --strategy $strategy --distribution $distribution \
                    --percentage_noisy_clients $percentages --fit_clients $fit_clients --seed $seed;

                    flwr run;
            done
        done
    done
done