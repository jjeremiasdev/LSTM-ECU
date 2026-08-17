import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def crear_ventanas(data_scaled, lookback=12):
    """
    Transforma la serie de tiempo bidimensional en un tensor 3D 
    (muestras, pasos_de_tiempo, caracteristicas)
    """
    X, y = [], []
    # La columna 0 debe ser siempre el target (empleo_adecuado)
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i, :])
        y.append(data_scaled[i, 0]) 
    return np.array(X), np.array(y)

def build_features():
    print("Iniciando Feature Engineering...")
    
    # 1. Cargar el dataset consolidado
    df = pd.read_csv('data/raw/dataset.csv', parse_dates=['fecha'])
    
    # 2. Estacionarización de variables exógenas (Tasas de Variación Mensual)
    df['ipc_pct'] = df['ipc'].pct_change().fillna(0) * 100
    df['wti_pct'] = df['wti'].pct_change().fillna(0) * 100
    
    # Codificación Cíclica Temporal (Seno y Coseno)
    df['mes'] = df['fecha'].dt.month
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    
    # Ordenar columnas para que el target (empleo_adecuado) sea siempre la primera (indice 0)
    features = ['empleo_adecuado', 'ipc_pct', 'wti_pct', 'es_interpolado', 'mes_sin', 'mes_cos']
    df_modelo = df[['fecha'] + features].copy()
    
    # 3. Partición de Datos (Train: 2015-2023, Val: 2024, Test: 2025)
    train_mask = df_modelo['fecha'] < '2024-01-01'
    val_mask = (df_modelo['fecha'] >= '2024-01-01') & (df_modelo['fecha'] < '2025-01-01')
    test_mask = df_modelo['fecha'] >= '2025-01-01'
    
    # Extraemos solo los valores numericos para escalar
    # Cuidado: ajustamos el scaler SOLO con los datos de entrenamiento para no "espiar" el futuro (Data Leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Separamos en arrays crudos
    train_data = df_modelo.loc[train_mask, features].values
    val_data = df_modelo.loc[val_mask, features].values
    test_data = df_modelo.loc[test_mask, features].values
    
    # Entrenamos el scaler solo con Train y transformamos todos
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    # 4. Creación de la Ventana Deslizante (Tensores 3D) T=12
    lookback = 12
    
    # IMPORTANTE: Para que la validacion y test tengan datos completos desde su primer mes, 
    # necesitamos que "tomen prestados" los ultimos 12 meses del bloque anterior para su primer calculo.
    val_data_extended = np.vstack((train_scaled[-lookback:], val_scaled))
    test_data_extended = np.vstack((val_scaled[-lookback:], test_scaled))
    
    X_train, y_train = crear_ventanas(train_scaled, lookback)
    X_val, y_val = crear_ventanas(val_data_extended, lookback)
    X_test, y_test = crear_ventanas(test_data_extended, lookback)
    
    # 5. Guardar los tensores para que la LSTM los consuma directamente
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_test.npy', y_test)
    
    # Guardamos los metadatos del Scaler si queremos revertir la prediccion a % reales despues
    import joblib
    joblib.dump(scaler, 'data/processed/scaler.save')
    
    print("¡Tensores creados con éxito!")
    print(f"Shape X_train (Tensores): {X_train.shape} -> (Muestras, T=12, Variables=6)")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Shape X_val: {X_val.shape}")
    print(f"Shape X_test: {X_test.shape}")

if __name__ == '__main__':
    build_features()
