import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Charger le dataset nettoyé
dataset_path = 'dataset_nettoye.csv'
df = pd.read_csv(dataset_path)

# Séparer les features (X) et la variable cible (y)
X = df.drop(['isFraud'], axis=1)
y = df['isFraud']

# Gérer les valeurs manquantes pour les colonnes numériques
numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

# Encoder la variable catégorielle "type"
label_encoder_type = LabelEncoder()
X['type'] = label_encoder_type.fit_transform(X['type'])

# Encoder les variables catégorielles "nameOrig" et "nameDest"
label_encoder_name = LabelEncoder()
X['nameOrig'] = label_encoder_name.fit_transform(X['nameOrig'])
X['nameDest'] = label_encoder_name.fit_transform(X['nameDest'])

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

dump(model, "modele.joblib")
print(f"Précision du modèle : {accuracy:.2f}")
print("Rapport de classification :\n", classification_rep)