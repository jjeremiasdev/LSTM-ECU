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
import copy
import pandas as pd

from src.models.architecture import LSTMModel, GRUModel, BiLSTMModel, AttentionLSTMModel

def inverse_transform_target(scaled_target, scaler):
    dummy = np.zeros((len(scaled_target), 6))
    dummy[:, 0] = scaled_target.ravel()
    return scaler.inverse_transform(dummy)[:, 0]

def calculate_smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def train_single_model(model_name, model_cls, X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test, scaler):
    print(f"\n---> Entrenando {model_name}...")
    torch.manual_seed(42)
    np.random.seed(42)

    features = X_train_t.shape[2]
    model = model_cls(input_size=features, hidden_size=64, output_size=1, dropout=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
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
            print(f"[{model_name}] Early stopping en época {epoch+1} (val_loss: {best_val_loss:.4f})")
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

    # Guardar pesos
    os.makedirs(f'models/{model_name.lower()}', exist_ok=True)
    torch.save(model.state_dict(), f'models/{model_name.lower()}/model.pth')
    np.save(f'data/processed/y_pred_{model_name.lower()}.npy', y_pred_real)

    return {"Modelo": model_name, "RMSE": rmse, "MAE": mae, "sMAPE (%)": smape}

def benchmark_dl_models():
    print("==============================================")
    print("   BENCHMARKING DE ARQUITECTURAS DE DEEP LEARNING   ")
    print("==============================================")

    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    scaler = joblib.load('data/processed/scaler.save')

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    architectures = [
        ("LSTM", LSTMModel),
        ("GRU", GRUModel),
        ("BiLSTM", BiLSTMModel),
        ("AttentionLSTM", AttentionLSTMModel)
    ]

    results = []
    for name, cls in architectures:
        res = train_single_model(name, cls, X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test, scaler)
        results.append(res)

    df_results = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n==============================================")
    print("  RESULTADOS COMPARATIVOS EN TEST SET (2025)  ")
    print("==============================================")
    print(df_results.to_string(index=False))
    print("==============================================\n")

    os.makedirs('reports', exist_ok=True)
    df_results.to_csv('reports/deep_learning_benchmark.csv', index=False)

if __name__ == '__main__':
    benchmark_dl_models()
