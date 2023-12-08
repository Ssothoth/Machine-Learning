import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charger le dataset depuis le fichier CSV
dataset_path = 'dataset_cleaned.csv'
df = pd.read_csv(dataset_path)

# Mapping transaction types to numerical values
df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[4, 5, 2, 3, 1], inplace=True)

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Initialiser le modèle K-means avec le nombre de clusters souhaité
num_clusters = 3  # Vous pouvez ajuster le nombre de clusters selon vos besoins
model = KMeans(n_clusters=num_clusters, random_state=42)

# Effectuer le clustering
df['cluster'] = model.fit_predict(X_scaled)

# Visualiser les clusters (vous pouvez ajuster cela en fonction de vos colonnes)
plt.scatter(df['amount'], df['oldbalanceOrg'], c=df['cluster'], cmap='viridis')
plt.title('Clusters K-means')
plt.xlabel('Amount')
plt.ylabel('Old Balance Org')
plt.show()
accuracy = model.score(X_scaled)
print(f"Précision du modèle : {accuracy}")