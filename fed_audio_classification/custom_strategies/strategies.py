from typing import List, Tuple, Dict, Any
import flwr as fl
from flwr.common import FitRes

from numpy import ndarray as NDArrays
from flwr.server.client_proxy import ClientProxy
import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


class FedSNR(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[object, Dict[str, Any]]:
        
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        
        # Custom aggregation logic
        aggregated_ndarrays = self.aggregate_inplace(results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        print("FedSNR Strategy")
        return parameters_aggregated, {}
    
    def aggregate_inplace(self, results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
        # Extract SNR values (ensure metric name matches what clients report)
        snr_values = [fit_res.metrics["mean_snr"] for _, fit_res in results] # che per ora sono quelli medi
        total_snr = sum(snr_values)
        
        # Compute scaling factors based purely on SNR
        scale_factors = [snr / total_snr for snr in snr_values]
        
        # Convert parameters to NDArrays
        parameters_array = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        
        # Weighted average using SNR scaling factors
        aggregated_ndarrays = [
            np.zeros_like(layer, dtype=np.float64) for layer in parameters_array[0]
        ]
        for client_weights, scale in zip(parameters_array, scale_factors):
            for i, layer in enumerate(client_weights):
                aggregated_ndarrays[i] += layer * scale
                
        return aggregated_ndarrays
    

class FedSNRCS(fl.server.strategy.FedAvg):
    def __init__(self, k: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k  # numero massimo di client da selezionare ad ogni round

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[object, Dict[str, Any]]:
        
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # 1. Seleziona i top-k client in base allo SNR
        selected_results = self.select_top_k(results, self.k)

        # 2. Aggrega solo i client selezionati, pesando per SNR
        aggregated_ndarrays = self.aggregate_inplace(selected_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        print(f"FedSNR Strategy (round {rnd}): selezionati {len(selected_results)} client su {len(results)}")
        return parameters_aggregated, {}

    def select_top_k(
        self, results: List[Tuple[ClientProxy, FitRes]], k: int
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Ordina i client in base al mean_snr e restituisce i top-k."""
        sorted_results = sorted(
            results,
            key=lambda r: r[1].metrics.get("mean_snr", 0.0),
            reverse=True,
        )
        return sorted_results[:k]

    def aggregate_inplace(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> NDArrays:
        """Aggregazione pesata per mean_snr dei client selezionati."""
        snr_values = [res.metrics.get("mean_snr", 1e-6) for _, res in results]
        total_snr = sum(snr_values)
        if total_snr == 0:
            # fallback: media semplice
            snr_values = [1.0 for _ in snr_values]
            total_snr = len(snr_values)

        # calcola pesi normalizzati
        scale_factors = [snr / total_snr for snr in snr_values]

        # parametri in numpy arrays
        parameters_array = [
            parameters_to_ndarrays(res.parameters) for _, res in results
        ]

        # aggregazione pesata
        aggregated_ndarrays = [
            np.zeros_like(layer, dtype=np.float64) for layer in parameters_array[0]
        ]
        for client_weights, scale in zip(parameters_array, scale_factors):
            for i, layer in enumerate(client_weights):
                aggregated_ndarrays[i] += layer * scale

        return aggregated_ndarrays
    


import numpy as np
import flwr as fl
from flwr.common import NDArrays, FitRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Any


class FedSNRSelection(fl.server.strategy.FedAvg):
    def __init__(self, fraction_best: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.fraction_best = fraction_best  # frazione top-k client da usare
        self.client_snr_history: Dict[str, float] = {}  # salva ultimo mean_snr visto per client_id

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        """Seleziona solo la frazione migliore di client in base al mean_snr noto."""
        all_clients = list(client_manager.all().values())

        if server_round == 1:
            # Prende la config dal server (già definita)
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}

            # 👇 Qui costruiamo un FitIns con i parametri e la config
            fit_ins = FitIns(parameters, config)

            return [(client, fit_ins) for client in all_clients]
 
        # Ordina i client in base allo SNR noto (default 0.0 se non visto ancora)
        sorted_clients = sorted(
            all_clients,
            key=lambda c: self.client_snr_history.get(c.cid, 0.0),
            reverse=True,
        )

        # Numero di client da selezionare
        k = max(1, int(len(sorted_clients) * self.fraction_best))
        selected_clients = sorted_clients[:k]

        # 🔹 Logga tutti i client e i loro SNR
        print(f"[Round {server_round}] Tutti i client disponibili e SNR:")
        for c in all_clients:
            snr = self.client_snr_history.get(c.cid, 0.0)
            print(f"  Client {c.cid}: {snr}")

        # 🔹 Logga quali sono stati scelti
        print(f"[Round {server_round}] Client selezionati per il fit:")
        for c in selected_clients:
            snr = self.client_snr_history.get(c.cid, 0.0)
            print(f"  Client {c.cid}: {snr}")


        # Prende la config dal server (già definita)
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}

        # 👇 Qui costruiamo un FitIns con i parametri e la config
        fit_ins = FitIns(parameters, config)

        return [(client, fit_ins) for client in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[object, Dict[str, Any]]:
        
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # 🔹 aggiorna dizionario con i nuovi mean_snr dai metrics
        for client, fit_res in results:
            if "mean_snr" in fit_res.metrics:
                self.client_snr_history[client.cid] = fit_res.metrics["mean_snr"]

        # Aggregazione personalizzata pesata con SNR
        aggregated_ndarrays = self.aggregate_inplace(results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        print(f"FedSNR Strategy (round {server_round})")
        return parameters_aggregated, {}
    
    def aggregate_inplace(self, results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
        # Estrai valori di mean_snr
        snr_values = [fit_res.metrics.get("mean_snr", 0.0) for _, fit_res in results]
        total_snr = sum(snr_values)
        
        # Evita divisione per zero
        if total_snr == 0:
            scale_factors = [1.0 / len(snr_values)] * len(snr_values)
        else:
            scale_factors = [snr / total_snr for snr in snr_values]
        
        # Converti parametri a NDArrays
        parameters_array = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        
        # Media pesata
        aggregated_ndarrays = [
            np.zeros_like(layer, dtype=np.float64) for layer in parameters_array[0]
        ]
        for client_weights, scale in zip(parameters_array, scale_factors):
            for i, layer in enumerate(client_weights):
                aggregated_ndarrays[i] += layer * scale
                
        return aggregated_ndarrays