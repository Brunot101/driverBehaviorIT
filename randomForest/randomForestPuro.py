import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold
import json

df = pd.read_csv("../trafficDriveFull.csv", sep=",")
#Colocar desvio padrão em todas as métricas de avaliação
colluns = [
    'AltitudeVariation',
    'VehicleSpeedInstantaneous',
    'VehicleSpeedAverage',
    'VehicleSpeedVariance',
    'VehicleSpeedVariation',
    'LongitudinalAcceleration',
    'EngineLoad',
    'ManifoldAbsolutePressure',
    'EngineRPM',
    'MassAirFlow',
    'IntakeAirTemperature',
    'VerticalAcceleration',
    'FuelConsumptionAverage'
    ]

df = df.dropna()


for col in colluns:
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)

X1 = df[colluns]  
y = df["drivingStyle"]

rf1 = RandomForestClassifier(n_estimators=100, random_state=42)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


scores = cross_validate(
    rf1, X1, y, cv=cv,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
)

print("=== Cenario 1: Features originais com 5-Fold Cross-Validation ===")
print("Acurácia média:", scores["test_accuracy"].mean())
print("Precisão média:", scores["test_precision_macro"].mean())
print("Recall médio:", scores["test_recall_macro"].mean())
print("F1-score médio:", scores["test_f1_macro"].mean())


results = {
    "Cenario 1 - Features originais": {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std" : scores['test_accuracy'].std(),
        "precision_mean": scores["test_precision_macro"].mean(),
        "precision_std": scores["test_precision_macro"].std(),
        "recall_mean": scores["test_recall_macro"].mean(),
        "recall_std": scores["test_recall_macro"].std(),
        "f1_mean": scores["test_f1_macro"].mean(),
        "f1_std": scores["test_f1_macro"].std(),
    }
}


with open("featuresOriginais.json", "w") as f:
    json.dump(results, f, indent=4)

