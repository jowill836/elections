import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement des  données depuis votre fichier XLS
data = pd.read_excel("donneesTraites/E.xlsx")

# Division des données en caractéristiques (X) et cible (y)
X = data[["chomage", "Inflation", "PIB"]]
y = data["Orientation"]

# Divisions des données en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Liste des modèles à tester
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Entraînement et évaluation de  chaque modèle
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Modèle : {model_name}")
    print(f"Précision du modèle : {accuracy:.4f}")
    print(f"Rapport de classification :\n{report}")
    print("=" * 50)
