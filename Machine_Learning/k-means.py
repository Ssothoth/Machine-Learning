import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charger le dataset depuis le fichier CSV
dataset_path = 'dataset.csv'
df = pd.read_csv(dataset_path)

# Remplacer les valeurs manquantes par la moyenne des colonnes respectives
df = df.fillna(df.mean())

# Assurer que la colonne 'step' est de type entier
df['step'] = df['step'].astype(int)

# Assurer que la colonne 'type' est de type chaîne de caractères
df['type'] = df['type'].astype(str)

# Assurer que les colonnes 'nameOrig' et 'nameDest' sont de type chaîne de caractères
df['nameOrig'] = df['nameOrig'].astype(str)
df['nameDest'] = df['nameDest'].astype(str)

# Encoder les colonnes catégorielles avec one-hot encoding
df_encoded = pd.get_dummies(df, columns=['type', 'nameOrig', 'nameDest'])

# Assurer que les colonnes 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest' sont de type flottant
df_encoded['amount'] = df_encoded['amount'].astype(float)
df_encoded['oldbalanceOrg'] = df_encoded['oldbalanceOrg'].astype(float)
df_encoded['newbalanceOrig'] = df_encoded['newbalanceOrig'].astype(float)
df_encoded['oldbalanceDest'] = df_encoded['oldbalanceDest'].astype(float)
df_encoded['newbalanceDest'] = df_encoded['newbalanceDest'].astype(float)

# Assurer que les colonnes 'isFraud' et 'isFlaggedFraud' sont de type flottant
df_encoded['isFraud'] = df_encoded['isFraud'].astype(float)
df_encoded['isFlaggedFraud'] = df_encoded['isFlaggedFraud'].astype(float)

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Initialiser le modèle K-means avec le nombre de clusters souhaité
num_clusters = 3  # Vous pouvez ajuster le nombre de clusters selon vos besoins
model = KMeans(n_clusters=num_clusters, random_state=42)

# Effectuer le clustering
df_encoded['cluster'] = model.fit_predict(X_scaled)

# Visualiser les clusters (vous pouvez ajuster cela en fonction de vos colonnes)
plt.scatter(df_encoded['amount'], df_encoded['oldbalanceOrg'], c=df_encoded['cluster'], cmap='viridis')
plt.title('Clusters K-means')
plt.xlabel('Amount')
plt.ylabel('Old Balance Org')
plt.show()
