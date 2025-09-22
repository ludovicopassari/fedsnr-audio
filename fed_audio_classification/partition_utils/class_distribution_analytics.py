import pandas as pd
import matplotlib.pyplot as plt
from config import PARTITIONING_METADATA_DIRICHLET , PARTITIONING_METADATA_IID

# Leggi il CSV
csv_file = PARTITIONING_METADATA_DIRICHLET # sostituisci con il percorso del tuo CSV
df = pd.read_csv(csv_file)

# Rimuovi le righe con partition_id == -1
df = df[df['partition_id'] != -1]

# Controlla i primi record
print(df.head())

# Raggruppa per 'partition_id' e 'class' e conta le occorrenze
class_distribution = df.groupby(['partition_id', 'class']).size().unstack(fill_value=0)

# Mostra la tabella di distribuzione
print(class_distribution)

# Plot della distribuzione
class_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title("Distribuzione delle classi per client")
plt.xlabel("Client ID")
plt.ylabel("Numero di campioni")
plt.xticks(rotation=0)
plt.legend(title="Class")
plt.tight_layout()
plt.show()
