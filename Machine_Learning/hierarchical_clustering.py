import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Les données que vous avez fournies
algorithmes = ['XG Boost', 'Random Forest', 'Logistic Regression', 'Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Optimise Gradient Boosting Regressor', 'LightGBM Regressor', 'Decision Tree']
precision = [0.9997721064592888, 0.9997225985521687, 0.998313587798737, 0.998840886301555, 0.9988259553454395, 0.9988416721413506, 0.9991693673360974, 0.999706, 0.9997225985521687]
fraudes_predites = [1397, 1292, 672, 143, 126, 143, 581, 1266, 1415]
fraudes_reelles = 1620  # Nombre de fraudes réelles
total_donnes = 1272524  # Nombre total de données dans l'échantillon

# Calcul des métriques supplémentaires
sensibilite = fraudes_reelles / total_donnes  # À ajuster selon votre définition spécifique
specificite = [1 - (fraudes_predites[i] / (total_donnes - fraudes_reelles)) for i in range(len(fraudes_predites))]  # Ajusté pour itérer sur la liste
f1_score = 2 * (precision * sensibilite) / (precision + sensibilite)

# Création d'un DataFrame
df = pd.DataFrame({'Algorithmes': algorithmes, 'Précision': precision, 'Sensibilité': sensibilite, 'Spécificité': specificite, 'F1-Score': f1_score, 'Fraudes Prédites': fraudes_predites})

# Paramètres du graphique
fig, ax1 = plt.subplots(figsize=(10, 6))

# Barres pour la Précision
color = 'tab:blue'
ax1.set_xlabel('Algorithmes')
ax1.set_ylabel('Précision', color=color)
ax1.bar(df['Algorithmes'], df['Précision'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Création d'une seconde axe y pour Fraudes Prédites
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Fraudes Prédites', color=color)
ax2.plot(df['Algorithmes'], df['Fraudes Prédites'], color=color, marker='o', label='Fraudes Prédites')
ax2.axhline(fraudes_reelles, linestyle='--', color='green', label='Fraudes Réelles 0.13%')
ax2.tick_params(axis='y', labelcolor=color)

# Ajout d'une troisième axe y pour les métriques supplémentaires
ax3 = ax1.twinx()
color = 'tab:purple'
ax3.spines['right'].set_position(('outward', 60))  # Ajustement de la position de l'axe
ax3.set_ylabel('Métriques Supplémentaires', color=color)
ax3.plot(df['Algorithmes'], df['Sensibilité'], label='Sensibilité', color='orange', marker='o')
ax3.plot(df['Algorithmes'], df['Spécificité'], label='Spécificité', color='purple', marker='o')
ax3.plot(df['Algorithmes'], df['F1-Score'], label='F1-Score', color='green', marker='o')
ax3.tick_params(axis='y', labelcolor=color)

# Titre et affichage du graphique
plt.title('Comparaison des Algorithmes de Détection de Fraudes')
plt.legend(loc='upper left')
plt.show()
