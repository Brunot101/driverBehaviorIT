import pandas as pd
import numpy as np
import json
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping


pasta = "traficDrive"
arquivo = "trafficDriveFull60_dx5_with_y.csv"
window_size = 10  

df = pd.read_csv("../" + pasta + "/" + arquivo, sep=",")

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

df = df.dropna().reset_index(drop=True)

X = df[colluns].values
y = df["drivingStyle"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_encoded, window_size)
y_seq_categorical = to_categorical(y_seq)

print(f"Formato de entrada para LSTM: {X_seq.shape}  (amostras, timesteps, features)")


n_splits = 5
epochs = 50
batch_size = 32
results = []

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold = 1

for train_idx, test_idx in skf.split(X_seq, y_seq):
    print(f"\nTreinando Fold {fold}/{n_splits}...")

    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq_categorical[train_idx], y_seq_categorical[test_idx]

    model = Sequential([
        LSTM(64, input_shape=(window_size, X_seq.shape[2]), return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(y_seq_categorical.shape[1], activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.2,
              verbose=0,
              callbacks=[early_stop])

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Fold {fold} -> Acurácia: {acc:.4f}, Precisão: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    results.append({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    fold += 1


accuracy_mean = np.mean([r["accuracy"] for r in results])
precision_mean = np.mean([r["precision"] for r in results])
recall_mean = np.mean([r["recall"] for r in results])
f1_mean = np.mean([r["f1"] for r in results])

accuracy_std = np.std([r["accuracy"] for r in results])
precision_std = np.std([r["precision"] for r in results])
recall_std = np.std([r["recall"] for r in results])
f1_std = np.std([r["f1"] for r in results])

padrao = r'Full(\d+)_dx(\d+)'
match = re.search(padrao, arquivo)
window = match.group(1) if match else "NA"
dx_valor = match.group(2) if match else "NA"

print("\n=== Cenário LSTM com janelas temporais ===")
print(f"Acurácia média: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
print(f"Precisão média: {precision_mean:.4f} ± {precision_std:.4f}")
print(f"Recall médio:    {recall_mean:.4f} ± {recall_std:.4f}")
print(f"F1-score médio:  {f1_mean:.4f} ± {f1_std:.4f}")


results_json = {
    "Cenario LSTM - Janelas temporais": {
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "dx": dx_valor,
        "window": window,
        "timesteps": window_size
    }
}

with open(f"LSTM_W{window}_dx_{dx_valor}_timesteps{window_size}.json", "w") as f:
    json.dump(results_json, f, indent=4)

print("\nResultados salvos com sucesso!")
