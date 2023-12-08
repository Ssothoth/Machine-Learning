import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Charger le dataset nettoyé
dataset_path = 'dataset_cleaned.csv'
df = pd.read_csv(dataset_path)

# Mapping transaction types to numerical values
df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[4, 5, 2, 3, 1], inplace=True)

# Séparer les features (X) et la variable cible (y)
X = df.drop(['isFraud'], axis=1)
y = df['isFraud']

# Diviser le dataset en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2, n_jobs=-1)

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Prédire la fraude sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
accuracy = model.score(X_test, y_test)
#dump(model, "modele.joblib")
print(f"Précision du modèle : {accuracy:.2f}")
print("Rapport de classification :\n", classification_rep)
prediction = model.predict(X_test)
prediction = (prediction > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, prediction)
print(f"Confusion matrix :\n{conf_matrix}")