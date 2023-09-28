import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Charger votre fichier Excel
data = pd.read_excel("donneesTraites/E.xlsx")

# Diviser les données en caractéristiques (X) et la variable cible (y)
X = data[["chomage", "Inflation", "PIB"]]
y = data["Orientation"]

#Ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Création  d'un modèle RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Entraînement du modele
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation des performances du modèle
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Précision du modèle : {accuracy}")
print("Rapport de classification :")
print(classification_rep)

# Test de prediction

new_data = {
    "chomage": 7.0,  # Remplacez par la valeur de chômage que vous souhaitez prédire
    "Inflation": 3.0,  # Remplacez par la valeur d'inflation que vous souhaitez prédire
    "PIB": 50000.0  # Remplacez par la valeur de PIB que vous souhaitez prédire
}

new_data_df = pd.DataFrame([new_data])

predicted_orientation = model.predict(new_data_df)

print("Prédiction de l'orientation politique : ", predicted_orientation[0])

#Test de prediction sur des annees

# données historiques pour les caractéristiques (chômage, inflation, PIB)
historical_data = pd.read_excel("donneesTraites/E.xlsx")


# Préparez des données pour les années à venir (1 an, 2 ans, 3 ans)
# Remplacez ces valeurs par celles que vous souhaitez prédire
future_years = [2027, 2032, 2037]
future_data = pd.DataFrame({
    "chomage": [7.0, 6.5, 6.0],  # Exemple de taux de chômage prévu pour les années à venir
    "Inflation": [2.0, 2.2, 2.5],  # Exemple de taux d'inflation prévu pour les années à venir
    "PIB": [45000.0, 47000.0, 49000.0]  # Exemple de PIB prévu pour les années à venir
})

# Prédire l'orientation politique pour les années à venir
future_predictions = model.predict(future_data)

# Créez un graphique pour visualiser les prédictions
plt.figure(figsize=(10, 6))
plt.plot(historical_data["Annee"], historical_data["Orientation"], label="Historique", marker='o')
plt.plot(future_years, future_predictions, label="Prédictions futures", marker='x', linestyle='--')
plt.xlabel("Année")
plt.ylabel("Orientation politique")
plt.title("Prédictions futures de l'orientation politique")
plt.legend()
plt.grid(True)
plt.show()
