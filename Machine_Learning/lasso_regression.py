import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix

# Charger le dataset depuis le fichier CSV
dataset_path = 'dataset_cleaned.csv'
df = pd.read_csv(dataset_path)

# Mapping transaction types to numerical values
df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[4, 5, 2, 3, 1], inplace=True)

# Encoder les colonnes catégorielles avec one-hot encoding
df_encoded = pd.get_dummies(df, columns=['type'])

# Séparer les données en features (X) et la variable cible (y)
X = df_encoded.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df_encoded['isFraud']

# Diviser le dataset en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle Lasso
alpha_value = 1.0  # Vous pouvez ajuster la valeur d'alpha selon vos besoins
model = Lasso(alpha=alpha_value)

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Prédire les valeurs sur l'ensemble de test
y_pred = model.predict(X_test)

# Evaluer la performance du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
accuracy = model.score(X_test, y_test)
print(f"Précision du modèle : {accuracy}")
prediction = model.predict(X_test)
prediction = (prediction > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, prediction)
print(f"Confusion matrix :\n{conf_matrix}")