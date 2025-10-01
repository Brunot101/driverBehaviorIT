import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold
import json
import re

arquivo = "trafficDriveFull60_dx5_with_y.csv"
df = pd.read_csv("../traficDrive/" + arquivo, sep=",")

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
    'FuelConsumptionAverage',
]

df = df.dropna()

X1 = df[colluns]  
y = df["drivingStyle"]


dt1 = DecisionTreeClassifier(random_state=42, max_depth=None)  


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    dt1, X1, y, cv=cv,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
)

print("=== Cenário 1: Features originais (Decision Tree) com 5-Fold Cross-Validation ===")
print("Acurácia média:", scores["test_accuracy"].mean())
print("Precisão média:", scores["test_precision_macro"].mean())
print("Recall médio:", scores["test_recall_macro"].mean())
print("F1-score médio:", scores["test_f1_macro"].mean())

# Extraindo janela e dx do nome do arquivo
padrao = r'Full(\d+)_dx(\d+)'
match = re.search(padrao, arquivo)

if match:
    window = match.group(1)
    dx_valor = match.group(2)
else:
    print("Padrão não encontrado.")
    window, dx_valor = "NA", "NA"

# Salvando resultados em JSON
results = {
    "Cenario 3 - Decision Tree com teoria da informacao por linha": {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "precision_mean": scores["test_precision_macro"].mean(),
        "recall_mean": scores["test_recall_macro"].mean(),
        "f1_mean": scores["test_f1_macro"].mean(),
        "dx": dx_valor,
        "window": window
    } 
}

with open("featuresOriginais.json", "w") as f:
    json.dump(results, f, indent=4)
