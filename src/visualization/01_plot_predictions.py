import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.dates as mdates

def plot_predictions():
    # Cargar datos
    y_test_real = np.load('data/processed/y_test_real.npy')
    y_pred_real = np.load('data/processed/y_pred_real.npy')
    
    # Cargar fechas reales del final del dataset
    df = pd.read_csv('data/raw/dataset.csv', parse_dates=['fecha'])
    test_dates = df['fecha'].tail(len(y_test_real))
    
    # Configuración del gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(test_dates, y_test_real, label='Real (Observado)', color='#1f77b4', linewidth=2, marker='o')
    plt.plot(test_dates, y_pred_real, label='Predicción LSTM', color='#d62728', linewidth=2, linestyle='--', marker='x')
    
    # Formato de fechas en el Eje X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    # Detalles estéticos
    plt.title('Proyección de la Tasa de Empleo Adecuado: LSTM vs Realidad (Out-of-Sample)', fontsize=14, fontweight='bold')
    plt.xlabel('Fechas de Evaluación (Test Set)', fontsize=12)
    plt.ylabel('Tasa de Empleo Adecuado (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Guardar en el proyecto y en la carpeta de artefactos
    os.makedirs('reports/figures', exist_ok=True)
    project_path = 'reports/figures/lstm_predictions.png'
    artifact_path = r'C:\Users\Admin\.gemini\antigravity\brain\4e7ed046-b23a-442b-a4c9-542e895aa7cf\lstm_predictions.png'
    
    plt.savefig(project_path, dpi=300)
    plt.savefig(artifact_path, dpi=300)
    
    print(f"Gráfico guardado en: {project_path}")

if __name__ == '__main__':
    plot_predictions()
