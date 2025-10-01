import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold
import json

df = pd.read_csv("../traficDrive/trafficDriveFull60_dx5_with_y.csv", sep=",")

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

rf1 = RandomForestClassifier(n_estimators=100, random_state=42)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


scores = cross_validate(
    rf1, X1, y, cv=cv,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
)

print("=== Cenário 2: Features com teoria da informação por coluna com 5-Fold Cross-Validation ===")
print("Acurácia média:", scores["test_accuracy"].mean())
print("Precisão média:", scores["test_precision_macro"].mean())
print("Recall médio:", scores["test_recall_macro"].mean())
print("F1-score médio:", scores["test_f1_macro"].mean())


results = {
    "Cenario 2 - Features com teoria da informacao por coluna": {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "precision_mean": scores["test_precision_macro"].mean(),
        "recall_mean": scores["test_recall_macro"].mean(),
        "f1_mean": scores["test_f1_macro"].mean()
    }
}


with open("teoriaInformacaoColuna.json", "w") as f:
    json.dump(results, f, indent=4)
