import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold
import json
import re

pasta = "traficDrive"
arquivo = "trafficDriveFull60_dx5_with_y.csv"
df = pd.read_csv("../"+pasta+"/" + arquivo, sep=",")


colluns = [
    'AltitudeVariation',
    'AltitudeVariation_entropy',
    'AltitudeVariation_complexity',
    'VehicleSpeedInstantaneous',
    'VehicleSpeedInstantaneous_complexity',
    'VehicleSpeedInstantaneous_entropy',
    'VehicleSpeedAverage',
    'VehicleSpeedAverage_entropy',
    'VehicleSpeedAverage_complexity',
    'VehicleSpeedVariance',
    'VehicleSpeedVariance_complexity',
    'VehicleSpeedVariance_entropy',
    'VehicleSpeedVariation',
    'VehicleSpeedVariation_entropy',
    'VehicleSpeedVariation_complexity',
    'LongitudinalAcceleration',
    'LongitudinalAcceleration_complexity',
    'LongitudinalAcceleration_entropy',
    'EngineLoad',
    'EngineLoad_entropy',
    'EngineLoad_complexity',
    'ManifoldAbsolutePressure',
    'ManifoldAbsolutePressure_complexity',
    'ManifoldAbsolutePressure_entropy',
    'EngineRPM',
    'EngineRPM_entropy',
    'EngineRPM_complexity',
    'MassAirFlow',
    'MassAirFlow_entropy',
    'MassAirFlow_complexity',
    'IntakeAirTemperature',
    'IntakeAirTemperature_entropy',
    'IntakeAirTemperature_complexity',
    'VerticalAcceleration',
    'VerticalAcceleration_entropy',
    'VerticalAcceleration_complexity',
    'FuelConsumptionAverage',
    'FuelConsumptionAverage_entropy',
    'FuelConsumptionAverage_complexity'
    ]

df = df.dropna()

X1 = df[colluns]  
y = df["drivingStyle"]

padrao = r'Full(\d+)_dx(\d+)'

match = re.search(padrao, arquivo)

if match:
    window = match.group(1)
    dx_valor = match.group(2)
else:
    print("Padrão não encontrado.")

rf1 = RandomForestClassifier(n_estimators=100, random_state=42)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


scores = cross_validate(
    rf1, X1, y, cv=cv,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
)

print("=== Cenário 2: Features com teoria da informação por coluna com 5-Fold Cross-Validation ===")
print(f"Acurácia média: {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
print(f"Precisão média: {scores['test_precision_macro'].mean():.4f} ± {scores['test_precision_macro'].std():.4f}")
print(f"Recall médio:    {scores['test_recall_macro'].mean():.4f} ± {scores['test_recall_macro'].std():.4f}")
print(f"F1-score médio:  {scores['test_f1_macro'].mean():.4f} ± {scores['test_f1_macro'].std():.4f}")

results = {
    "Cenario 2 - Features com teoria da informacao por coluna": {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std" : scores['test_accuracy'].std(),
        "precision_mean": scores["test_precision_macro"].mean(),
        "precision_std": scores["test_precision_macro"].std(),
        "recall_mean": scores["test_recall_macro"].mean(),
        "recall_std": scores["test_recall_macro"].std(),
        "f1_mean": scores["test_f1_macro"].mean(),
        "f1_std": scores["test_f1_macro"].std(),
        "dx": dx_valor,
        "window": window
    }
}


with open(pasta + window+"_dx_"+dx_valor+".json", "w") as f:
    json.dump(results, f, indent=4)
