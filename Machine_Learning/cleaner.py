import pandas as pd

# Charger le dataset depuis le fichier CSV
dataset_path = 'dataset.csv'
df = pd.read_csv(dataset_path)

# Supprimer les lignes avec des valeurs manquantes dans n'importe quelle colonne
df = df.dropna()

# Assurer que la colonne 'step' est de type entier
df['step'] = df['step'].astype(int)

# Assurer que la colonne 'type' est de type chaîne de caractères
df['type'] = df['type'].astype(str)

# Assurer que la colonne 'amount' est de type flottant
df['amount'] = df['amount'].astype(float)

# Assurer que les colonnes 'nameOrig' et 'nameDest' sont de type chaîne de caractères
df['nameOrig'] = df['nameOrig'].astype(str)
df['nameDest'] = df['nameDest'].astype(str)

# Assurer que les colonnes 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest' sont de type flottant
df['oldbalanceOrg'] = df['oldbalanceOrg'].astype(float)
df['newbalanceOrig'] = df['newbalanceOrig'].astype(float)
df['oldbalanceDest'] = df['oldbalanceDest'].astype(float)
df['newbalanceDest'] = df['newbalanceDest'].astype(float)

# Assurer que les colonnes 'isFraud' et 'isFlaggedFraud' sont de type flottant
df['isFraud'] = df['isFraud'].astype(float)
df['isFlaggedFraud'] = df['isFlaggedFraud'].astype(float)

# Enregistrer le dataset nettoyé dans un nouveau fichier CSV
df.to_csv('dataset_cleaned.csv', index=False)
