"""
Script de Simulación de 117 Experimentos Macroeconómicos Contrafácticos (Stress Testing)
Matriz Rejilla 13x9: 13 niveles de shock WTI (-30% a +30%) x 9 niveles de shock IPC (-1.0% a +3.0%)
Exporta los resultados a Excel, CSV y genera el Mapa de Calor 2D (Heatmap).
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.models.architecture import LSTMModel, GRUModel

def inverse_transform_target(scaled_target, scaler):
    dummy = np.zeros((len(scaled_target), 6))
    dummy[:, 0] = scaled_target.ravel()
    return scaler.inverse_transform(dummy)[:, 0]

def run_counterfactual_grid_experiments():
    print("==================================================")
    print("  EJECUTANDO 117 EXPERIMENTOS CONTRAFÁCTICOS (STRESS TESTING)  ")
    print("==================================================")

    # Cargar tensores y scaler
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    scaler = joblib.load('data/processed/scaler.save')

    features_dim = X_test.shape[2]
    
    # Cargar mejor modelo (GRU)
    model = GRUModel(input_size=features_dim, hidden_size=64, output_size=1, dropout=0.4)
    model_path = 'models/gru/model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        model = LSTMModel(input_size=features_dim, hidden_size=64, output_size=1, dropout=0.4)
        model.load_state_dict(torch.load('models/lstm/lstm_model.pth'))
    
    model.eval()

    # Definir la rejilla de umbrales realistas (13x9 = 117 experimentos)
    wti_shocks = np.linspace(-30, 30, 13)
    ipc_shocks = np.linspace(-1.0, 3.0, 9)

    resultados_lista = []
    matriz_pivote = np.zeros((len(ipc_shocks), len(wti_shocks)))

    exp_id = 1
    
    min_ipc, range_ipc = scaler.data_min_[1], scaler.data_range_[1]
    min_wti, range_wti = scaler.data_min_[2], scaler.data_range_[2]

    for i_idx, ipc_val in enumerate(ipc_shocks):
        for w_idx, wti_val in enumerate(wti_shocks):
            X_sim = X_test.copy()

            ipc_scaled = (ipc_val - min_ipc) / range_ipc if range_ipc != 0 else 0.5
            wti_scaled = (wti_val - min_wti) / range_wti if range_wti != 0 else 0.5

            X_sim[:, :, 1] = ipc_scaled
            X_sim[:, :, 2] = wti_scaled

            X_sim_t = torch.tensor(X_sim, dtype=torch.float32)
            with torch.no_grad():
                y_pred_scaled = model(X_sim_t).numpy()

            y_pred_real = inverse_transform_target(y_pred_scaled, scaler)

            promedio_empleo = np.mean(y_pred_real)
            std_empleo = np.std(y_pred_real)
            min_empleo = np.min(y_pred_real)
            max_empleo = np.max(y_pred_real)

            matriz_pivote[i_idx, w_idx] = promedio_empleo

            resultados_lista.append({
                "Exp_ID": f"EXP_{exp_id:03d}",
                "Shock_WTI_%": round(wti_val, 1),
                "Shock_IPC_%": round(ipc_val, 2),
                "Empleo_Adecuado_Promedio_%": round(promedio_empleo, 2),
                "Desviacion_Estandar": round(std_empleo, 2),
                "Empleo_Min_%": round(min_empleo, 2),
                "Empleo_Max_%": round(max_empleo, 2),
                "Escenario": "Estagflación" if (wti_val <= -15 and ipc_val >= 2.0) else (
                    "Bonanza" if (wti_val >= 15 and ipc_val <= 0.0) else "Simulación Estándar"
                )
            })
            exp_id += 1

    df_exps = pd.DataFrame(resultados_lista)

    # 1. Exportar CSV y Excel
    os.makedirs('reports', exist_ok=True)
    csv_path = 'reports/matriz_experimentos_stress_testing.csv'
    excel_path = 'reports/matriz_experimentos_stress_testing.xlsx'

    df_exps.to_csv(csv_path, index=False)

    df_pivote = pd.DataFrame(
        matriz_pivote,
        index=[f"IPC {ipc:+.1f}%" for ipc in ipc_shocks],
        columns=[f"WTI {wti:+.0f}%" for wti in wti_shocks]
    )

    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_pivote.to_excel(writer, sheet_name='Matriz_Pivote_2D')
            df_exps.to_excel(writer, sheet_name='Detalle_117_Experimentos', index=False)
        print(f"[OK] Registro de 117 experimentos guardado exitosamente en: {excel_path}")
    except Exception as e:
        print(f"[OK] Exportación básica CSV lista. Nota sobre Excel: {e}")

    # 2. Generar Mapa de Calor 2D (Heatmap)
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df_pivote,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        cbar_kws={'label': 'Tasa de Empleo Adecuado Estimada (%)'},
        linewidths=0.5
    )
    plt.title("Mapa de Calor 2D: Sensibilidad del Empleo Adecuado ante 117 Shocks de WTI e Inflación", fontsize=13, pad=15)
    plt.xlabel("Shock en la Variación del Precio del Petróleo WTI (Δ% WTI)", fontsize=11)
    plt.ylabel("Shock en la Tasa de Inflación Mensual (Δ% IPC)", fontsize=11)
    plt.tight_layout()

    heatmap_path = 'reports/figures/heatmap_sensibilidad_empleo.png'
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    print(f"[OK] Mapa de calor 2D generado en: {heatmap_path}")
    print(f"\nResumen de 117 Experimentos:")
    print(f"  - Empleo Promedio Mínimo Estimado: {df_exps['Empleo_Adecuado_Promedio_%'].min():.2f}% (Escenario de Estagflación)")
    print(f"  - Empleo Promedio Máximo Estimado: {df_exps['Empleo_Adecuado_Promedio_%'].max():.2f}% (Escenario de Bonanza Petrolera)")

if __name__ == '__main__':
    run_counterfactual_grid_experiments()
