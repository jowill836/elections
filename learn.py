import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
