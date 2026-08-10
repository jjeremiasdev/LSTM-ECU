import numpy as np
from scipy.stats import norm
import pandas as pd

def dm_test(actual, pred1, pred2, h=1, crit="MSE"):
    """
    Diebold-Mariano Test for predictive accuracy.
    actual: valores reales (numpy array)
    pred1: predicciones del modelo 1 (LSTM)
    pred2: predicciones del modelo 2 (Baseline)
    h: horizonte de prediccion
    crit: MSE o MAD (Mean Absolute Deviation)
    """
    e1 = actual - pred1
    e2 = actual - pred2
    
    # Calcular loss differential (d)
    if crit == "MSE":
        d = e1**2 - e2**2
    elif crit == "MAD":
        d = np.abs(e1) - np.abs(e2)
        
    d_mean = np.mean(d)
    
    # Autocovarianza para corregir dependencia serial
    T = float(len(d))
    gamma = []
    for lag in range(0, h):
        gamma.append(np.cov(d[0:int(T)-lag], d[lag:int(T)], bias=True)[0,1])
        
    v_hat = gamma[0]
    for lag in range(1, h):
        v_hat += 2.0 * gamma[lag]
        
    # Diebold-Mariano statistic
    dm_stat = d_mean / np.sqrt(v_hat / T)
    
    # p-value de una distribucion normal estandar (two-sided test)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

def run_tests():
    # Cargar datos (flattened to 1D arrays for easy computation)
    y_true = np.load('data/processed/y_test_real.npy').flatten()
    y_lstm = np.load('data/processed/y_pred_real.npy').flatten()
    y_xgb = np.load('data/processed/y_pred_xgboost.npy').flatten()
    y_sar = np.load('data/processed/y_pred_sarima.npy').flatten()
    y_var = np.load('data/processed/y_pred_var.npy').flatten()
    
    models = {
        "XGBoost": y_xgb,
        "VAR": y_var,
        "SARIMA": y_sar
    }
    
    print("=== TEST DE DIEBOLD-MARIANO (LSTM vs Baselines) ===")
    print("H0 (Hipótesis Nula): Ambos modelos tienen la misma precisión predictiva.")
    print("H1 (Hipótesis Alternativa): Las precisiones predictivas son significativamente diferentes.")
    print("Criterio: MSE (Mean Squared Error)\n")
    
    results = []
    
    for name, y_baseline in models.items():
        # dm_stat negativo significa que el modelo 1 (LSTM) tiene MENOR error que modelo 2 (Baseline)
        stat, p = dm_test(y_true, y_lstm, y_baseline, h=1, crit="MSE")
        
        # Interpretacion
        alpha = 0.05
        if p < alpha:
            conclusion = "Se rechaza H0. La LSTM es estadísticamente superior."
        else:
            conclusion = "No se rechaza H0. No hay diferencia estadísticamente significativa."
            
        print(f"LSTM vs {name}:")
        print(f"  Estadístico DM: {stat:.4f}")
        print(f"  P-Valor:        {p:.5f} (Alpha: {alpha})")
        print(f"  Conclusión:     {conclusion}\n")
        
        results.append({
            "Modelo Base": name,
            "DM Statistic": stat,
            "P-Value": p,
            "Es Superior?": "Sí" if p < alpha else "No"
        })
        
    # Guardar tabla
    df_results = pd.DataFrame(results)
    df_results.to_csv('reports/diebold_mariano_results.csv', index=False)
    
if __name__ == "__main__":
    run_tests()
