import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Charger le dataset depuis le fichier CSV
dataset_path = 'dataset_cleaned.csv'
df = pd.read_csv(dataset_path)

# Encoder les colonnes catégorielles avec one-hot encoding
df_encoded = pd.get_dummies(df, columns=['type'])

# Sélectionner les colonnes pertinentes pour l'algorithme Apriori
selected_columns = ['type_TRANSFER', 'type_CASH_OUT']
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