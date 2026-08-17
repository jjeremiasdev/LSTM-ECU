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

def inverse_transform_target(scaled_target, scaler):
    """
    Restaura la escala original para poder interpretar el error real en porcentaje.
    El scaler tiene 6 variables, el target es la columna 0.
    """
    dummy = np.zeros((len(scaled_target), 6))
    dummy[:, 0] = scaled_target.ravel()
    return scaler.inverse_transform(dummy)[:, 0]

def calculate_smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

from src.models.architecture import LSTMModel

def train_lstm():
    print("Cargando tensores...")
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    scaler = joblib.load('data/processed/scaler.save')

    # Convertir a tensores de PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    
    # Asegurar reproducibilidad de pesos iniciales
    torch.manual_seed(42)
    np.random.seed(42)

    T = X_train.shape[1]
    features = X_train.shape[2]
    
    print("Construyendo arquitectura LSTM...")
    model = LSTMModel(input_size=features, hidden_size=64, output_size=1, dropout=0.4)
    
    # El optimizador AdamW (desacopla el weight decay) según tu metodología
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("\nIniciando entrenamiento...")
    epochs = 150
    patience = 20
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

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
            
        train_loss /= len(train_loader.dataset)
        
        # Validación (eval function)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            
        # Early Stopping manual
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:03d}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping en epoch {epoch+1} (Mejor val_loss: {best_val_loss:.4f})")
            break

    # Restaurar los mejores pesos para probar
    model.load_state_dict(best_model_wts)

    print("\nRealizando predicciones sobre el Test Set...")
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy()

    # Invertir la escala para medir los errores en la unidad real (% de empleo adecuado)
    y_test_real = inverse_transform_target(y_test, scaler)
    y_pred_real = inverse_transform_target(y_pred_scaled, scaler)

    # Evaluar Métricas
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae = mean_absolute_error(y_test_real, y_pred_real)
    smape = calculate_smape(y_test_real, y_pred_real)

    print("\n==============================================")
    print("   RESULTADOS DE LA LSTM EN TEST (2025)  ")
    print("==============================================")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"sMAPE: {smape:.4f}%")
    print("==============================================\n")

    # Guardar artefactos
    os.makedirs('models/lstm', exist_ok=True)
    torch.save(model.state_dict(), 'models/lstm/lstm_model.pth')
    np.save('data/processed/y_test_real.npy', y_test_real)
    np.save('data/processed/y_pred_real.npy', y_pred_real)
    print("Modelo y arreglos predictivos guardados en disco.")

if __name__ == '__main__':
    train_lstm()
