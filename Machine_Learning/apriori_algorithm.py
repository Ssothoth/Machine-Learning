import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

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

# Sélectionner les colonnes pertinentes pour l'algorithme Apriori
selected_columns = ['type_TRANSFER', 'type_CASH_OUT', 'nameOrig_C', 'nameDest_C']
df_apriori = df_encoded[selected_columns]

# Convertir les données en format de transaction
transactions = df_apriori.values.tolist()

# Encoder les transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded_apriori = pd.DataFrame(te_ary, columns=te.columns_)

# Appliquer l'algorithme Apriori
frequent_itemsets = apriori(df_encoded_apriori, min_support=0.01, use_colnames=True)

# Afficher les règles d'association
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print(rules)