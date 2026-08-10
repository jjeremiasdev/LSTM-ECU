import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import os

def run_audit():
    os.makedirs('results', exist_ok=True)
    df = pd.read_csv('data/raw/dataset.csv', parse_dates=['fecha'])
    df = df.set_index('fecha')
    
    # 1. ADF Test
    print("--- Prueba de Dickey-Fuller Aumentada (ADF) ---")
    result = adfuller(df['empleo_adecuado'].dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for k, v in result[4].items():
        print(f"\t{k}: {v:.4f}")
    if result[1] < 0.05:
        print("Conclusion: La serie es ESTACIONARIA (Rechazamos H0).")
    else:
        print("Conclusion: La serie NO ES ESTACIONARIA (No rechazamos H0).")
            
    # 2. Graficas
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Grafica de Empleo Adecuado
    axes[0].plot(df.index, df['empleo_adecuado'], label='Empleo Adecuado (%)', color='blue')
    axes[0].set_title('Tasa de Empleo Adecuado')
    axes[0].legend()
    
    # Grafica de IPC
    axes[1].plot(df.index, df['ipc'], label='IPC', color='orange')
    axes[1].set_title('Índice de Precios al Consumidor (IPC)')
    axes[1].legend()
    
    # Grafica de WTI
    axes[2].plot(df.index, df['wti'], label='WTI', color='green')
    axes[2].set_title('Petróleo WTI')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results/audit_plot.png')
    print("\nGráficas guardadas en results/audit_plot.png")

if __name__ == "__main__":
    run_audit()
