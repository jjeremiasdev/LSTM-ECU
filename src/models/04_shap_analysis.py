import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import shap
import matplotlib.pyplot as plt

from src.models.architecture import LSTMModel

def shap_analysis():
    print("Iniciando SHAP Analysis...")
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    
    np.random.seed(42)
    # 50 muestras aleatorias del set de entrenamiento para usar de background (referencia)
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
    
    background_tensor = torch.tensor(background, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    features = X_train.shape[2]
    model = LSTMModel(input_size=features, hidden_size=64, output_size=1, dropout=0.4)
    model.load_state_dict(torch.load('models/lstm/lstm_model.pth'))
    model.eval()

    # GradientExplainer funciona matemáticamente mejor con las redes neuronales y backprop
    explainer = shap.GradientExplainer(model, background_tensor)
    
    print("Calculando valores SHAP (puede tardar un momento)...")
    shap_values = explainer.shap_values(test_tensor)
    
    # Manejar el formato de retorno de shap (a veces es lista, a veces tensor)
    if isinstance(shap_values, list):
        shap_values_3d = shap_values[0]
    else:
        shap_values_3d = shap_values
        
    # El tensor resultante es 4D por PyTorch (muestras, T=12, variables, 1).
    # Promediamos sobre el eje del tiempo (T) y aplastamos para obtener un resumen del peso global de cada variable (muestras, variables).
    shap_values_2d = np.mean(shap_values_3d, axis=1).squeeze()
    # También promediamos las observaciones del Test set para ponerlas de fondo en la gráfica
    test_tensor_2d = X_test.mean(axis=1)
    
    feature_names = ['Tasa Histórica (t-1)', 'Inflación (IPC)', 'Petróleo (WTI)', 'Es Interpolado', 'Mes (Sen)', 'Mes (Cos)']
    
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Gráfica Global (Bee Swarm / Summary)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_2d, test_tensor_2d, feature_names=feature_names, show=False)
    plt.title("Impacto Global SHAP en la LSTM (2025)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Gráfica de Barras (Magnitud Absoluta Promedio)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_2d, test_tensor_2d, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Importancia Promedio de las Variables Exógenas", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_bar.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("Análisis SHAP completado. Gráficas exportadas con éxito.")

if __name__ == '__main__':
    shap_analysis()
