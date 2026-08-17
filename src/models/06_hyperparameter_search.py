"""
Script de Búsqueda de Hiperparámetros (Grid Search) y Estudio de Ablación (Ablation Study)
para Validación Definitiva de Arquitecturas Recurrentes (LSTM y GRU) vs. Líneas Base.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import copy

from src.models.architecture import LSTMModel, GRUModel

def calculate_smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def inverse_transform_target(scaled_target, scaler):
    dummy = np.zeros((len(scaled_target), 6))
    dummy[:, 0] = scaled_target.ravel()
    return scaler.inverse_transform(dummy)[:, 0]

def train_eval_config(model_cls, X_train, y_train, X_val, y_val, X_test, y_test, scaler, hidden_size=64, lr=0.005, dropout=0.4):
    torch.manual_seed(42)
    np.random.seed(42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    features = X_train.shape[2]
    model = model_cls(input_size=features, hidden_size=hidden_size, output_size=1, dropout=dropout)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

    epochs = 150
    patience = 20
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy()

    y_test_real = inverse_transform_target(y_test, scaler)
    y_pred_real = inverse_transform_target(y_pred_scaled, scaler)

    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae = mean_absolute_error(y_test_real, y_pred_real)
    smape = calculate_smape(y_test_real, y_pred_real)

    return rmse, mae, smape

def run_grid_search_and_ablation():
    print("==================================================")
    print("   BÚSQUEDA DE HIPERPARÁMETROS Y ESTUDIO DE ABLACIÓN   ")
    print("==================================================")

    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    scaler = joblib.load('data/processed/scaler.save')

    results = []

    # 1. Variaciones de Neuronas Ocultas (Hidden Units) y LR en LSTM y GRU
    hidden_options = [32, 64, 128]
    lr_options = [0.001, 0.005, 0.01]

    for model_name, model_cls in [("LSTM", LSTMModel), ("GRU", GRUModel)]:
        for h in hidden_options:
            for lr in lr_options:
                rmse, mae, smape = train_eval_config(
                    model_cls, X_train, y_train, X_val, y_val, X_test, y_test, scaler,
                    hidden_size=h, lr=lr, dropout=0.4
                )
                results.append({
                    "Tipo": "Grid Search",
                    "Modelo": model_name,
                    "Hidden Units": h,
                    "Learning Rate": lr,
                    "Ablación / Config": f"Hidden={h}, LR={lr}",
                    "RMSE": rmse,
                    "MAE": mae,
                    "sMAPE (%)": smape
                })

    # 2. Estudio de Ablación de Variables
    # Experimento Ablación A: Sin Exógenas (Solo target t-1 y mes)
    # Seleccionar columnas 0 (target), 3 (es_interpolado), 4 (mes_sin), 5 (mes_cos)
    X_train_univariate = X_train[:, :, [0, 3, 4, 5]]
    X_val_univariate = X_val[:, :, [0, 3, 4, 5]]
    X_test_univariate = X_test[:, :, [0, 3, 4, 5]]

    for model_name, model_cls in [("LSTM", LSTMModel), ("GRU", GRUModel)]:
        rmse, mae, smape = train_eval_config(
            model_cls, X_train_univariate, y_train, X_val_univariate, y_val, X_test_univariate, y_test, scaler,
            hidden_size=64, lr=0.005, dropout=0.4
        )
        results.append({
            "Tipo": "Ablación",
            "Modelo": model_name,
            "Hidden Units": 64,
            "Learning Rate": 0.005,
            "Ablación / Config": "Sin Exógenas IPC/WTI",
            "RMSE": rmse,
            "MAE": mae,
            "sMAPE (%)": smape
        })

    # Experimento Ablación B: Sin Guardarraíl de Interpolación V_c
    X_train_noguard = X_train[:, :, [0, 1, 2, 4, 5]]
    X_val_noguard = X_val[:, :, [0, 1, 2, 4, 5]]
    X_test_noguard = X_test[:, :, [0, 1, 2, 4, 5]]

    for model_name, model_cls in [("LSTM", LSTMModel), ("GRU", GRUModel)]:
        rmse, mae, smape = train_eval_config(
            model_cls, X_train_noguard, y_train, X_val_noguard, y_val, X_test_noguard, y_test, scaler,
            hidden_size=64, lr=0.005, dropout=0.4
        )
        results.append({
            "Tipo": "Ablación",
            "Modelo": model_name,
            "Hidden Units": 64,
            "Learning Rate": 0.005,
            "Ablación / Config": "Sin Guardarraíl V_c (es_interpolado)",
            "RMSE": rmse,
            "MAE": mae,
            "sMAPE (%)": smape
        })

    df_res = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n==================================================")
    print("  RESULTADOS DE GRID SEARCH Y ABLACIÓN DE VARIABLES ")
    print("==================================================")
    print(df_res.to_string(index=False))
    print("==================================================\n")

    os.makedirs('reports', exist_ok=True)
    df_res.to_csv('reports/hyperparameter_ablation_results.csv', index=False)
    print("Tabla guardada en reports/hyperparameter_ablation_results.csv")

if __name__ == '__main__':
    run_grid_search_and_ablation()
