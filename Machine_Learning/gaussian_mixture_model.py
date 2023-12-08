import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Charger le dataset depuis le fichier CSV
dataset_path = 'dataset_cleaned.csv'
df = pd.read_csv(dataset_path)

# Mapping transaction types to numerical values
df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[4, 5, 2, 3, 1], inplace=True)

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Initialiser le modèle Gaussian Mixture avec le nombre de composantes souhaité (clusters)
num_components = 3
model = GaussianMixture(n_components=num_components, random_state=42)

# Ajuster le modèle aux données
model.fit(X_scaled)

# Prédire les clusters
df['cluster'] = model.predict(X_scaled)

# Visualiser les clusters (vous pouvez ajuster cela en fonction de vos colonnes)
plt.scatter(df['amount'], df['oldbalanceOrg'], c=df['cluster'], cmap='viridis')
plt.title('Clusters Gaussian Mixture Model')
plt.xlabel('Amount')
plt.ylabel('Old Balance Org')
plt.show()
accuracy = model.score(X_scaled)
print(f"Précision du modèle : {accuracy}")
prediction = model.predict(X_scaled)
prediction = (prediction > 0.5).astype(int)
conf_matrix = confusion_matrix(X_scaled, prediction)
print(f"Confusion matrix :\n{conf_matrix}")