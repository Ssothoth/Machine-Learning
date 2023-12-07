import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score

# Charger le dataset depuis le fichier CSV
dataset_path = 'dataset_fraud.csv'
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

# Séparer les données en features (X) et la variable cible (y)
X = df_encoded.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df_encoded['isFraud']

# Diviser le dataset en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #données en ensembles d'entraînement et de test, avec 80% des données pour 
                                                                                            #l'entraînement et 20% pour les tests.

# Initialiser le modèle de clustering hiérarchique
model = AgglomerativeClustering(n_clusters=2)  # spécifier nombre de clusters souhaités ?

# Ajuster le modèle sur l'ensemble d'entraînement
y_pred_train = model.fit_predict(X_train)

# Ajuster le modèle sur l'ensemble de test
y_pred_test = model.fit_predict(X_test)

# Evaluer la précision du modèle sur l'ensemble de test (Notez que cela est généralement utilisé pour l'évaluation des modèles de classification)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Précision du modèle : {accuracy}")
