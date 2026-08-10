import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def calculate_smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

print("--- INICIANDO MODELOS LÍNEA BASE ---")

# ==========================================
# 1. XGBoost (Requiere Tensores Aplanados)
# ==========================================
print("\nEntrenando XGBoost...")
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_test_real = np.load('data/processed/y_test_real.npy')
scaler = joblib.load('data/processed/scaler.save')

# Aplanar tensores 3D a 2D para XGBoost (N, T*Features)
X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train_2d, y_train)
y_pred_xgb_scaled = xgb_model.predict(X_test_2d)

# Invertir escala
dummy = np.zeros((len(y_pred_xgb_scaled), 6))
dummy[:, 0] = y_pred_xgb_scaled
y_pred_xgb_real = scaler.inverse_transform(dummy)[:, 0]

np.save('data/processed/y_pred_xgboost.npy', y_pred_xgb_real)

print(f"XGBoost RMSE:  {np.sqrt(mean_squared_error(y_test_real, y_pred_xgb_real)):.4f}")
print(f"XGBoost MAE:   {mean_absolute_error(y_test_real, y_pred_xgb_real):.4f}")
print(f"XGBoost sMAPE: {calculate_smape(y_test_real, y_pred_xgb_real):.4f}%")


# ==========================================
# Carga de Datos Crudos para SARIMA y VAR
# ==========================================
# Leemos los datos preprocesados pero sin escalar
df = pd.read_csv('data/raw/dataset.csv', parse_dates=['fecha']).set_index('fecha')
# df.index.freq = 'MS' # Opcional, pero suele fallar si las fechas no son exactamente el día 1

# Train/Test Split cronológico natural (sin ventana deslizante)
train_df = df[df.index < '2025-01-01']
test_df = df[df.index >= '2025-01-01']
steps = len(test_df)

# ==========================================
# 2. SARIMA (Univariado: solo empleo_adecuado)
# ==========================================
print("\nEntrenando SARIMA (Auto-Regresivo Integrado de Medias Móviles Estacional)...")
# Usamos orden tipico para series macroeconómicas con estacionalidad anual
sarima_model = SARIMAX(train_df['empleo_adecuado'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
y_pred_sarima_real = sarima_fit.forecast(steps=steps).values

np.save('data/processed/y_pred_sarima.npy', y_pred_sarima_real)

print(f"SARIMA RMSE:  {np.sqrt(mean_squared_error(y_test_real, y_pred_sarima_real)):.4f}")
print(f"SARIMA MAE:   {mean_absolute_error(y_test_real, y_pred_sarima_real):.4f}")
print(f"SARIMA sMAPE: {calculate_smape(y_test_real, y_pred_sarima_real):.4f}%")

# ==========================================
# 3. VAR (Multivariado Lineal)
# ==========================================
print("\nEntrenando VAR (Vectores Autorregresivos)...")
# VAR usa las 3 variables endógenas juntas
var_data = train_df[['empleo_adecuado', 'ipc', 'wti']]
# Diferenciamos para asegurar estacionariedad que VAR exige
var_data_diff = var_data.diff().dropna()
var_model = VAR(var_data_diff)
var_fit = var_model.fit(12)

# Prediccion recursiva
lag_order = var_fit.k_ar
last_obs = var_data_diff.values[-lag_order:]
forecast_diff = var_fit.forecast(y=last_obs, steps=steps)

# Revertir la diferenciación (Integrar)
y_pred_var_real = []
last_val = var_data['empleo_adecuado'].iloc[-1]
for val_diff in forecast_diff[:, 0]: # 0 es empleo_adecuado
    last_val = last_val + val_diff
    y_pred_var_real.append(last_val)

y_pred_var_real = np.array(y_pred_var_real)
np.save('data/processed/y_pred_var.npy', y_pred_var_real)

print(f"VAR RMSE:  {np.sqrt(mean_squared_error(y_test_real, y_pred_var_real)):.4f}")
print(f"VAR MAE:   {mean_absolute_error(y_test_real, y_pred_var_real):.4f}")
print(f"VAR sMAPE: {calculate_smape(y_test_real, y_pred_var_real):.4f}%")

print("\n--- TODOS LOS MODELOS ENTRENADOS ---")
