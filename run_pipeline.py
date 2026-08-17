"""
Pipeline Orquestador del Proyecto LSTM-ECU (Versión Integral de 4 Fases + 117 Experimentos)
Ejecuta la totalidad del flujo experimental:
1. Generación de Notebooks y Auditoría de Datos
2. Ingesta de Microdatos / Descarga WTI / Fusión de Series
3. Estacionarización de Exógenas e Ingeniería de Tensores 3D
4. Entrenamiento de Redes Recurrentes (LSTM, GRU, BiLSTM, Attention-LSTM)
5. Entrenamiento de Líneas Base (XGBoost, VAR, SARIMA)
6. Explicabilidad SHAP (XAI)
7. Inferencia Estadística de Diebold-Mariano
8. Simulación de 117 Experimentos Contrafácticos (Stress Testing & Heatmap)
9. Generación de Gráficos y Manuscrito Académico
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def run_step(description, command):
    print(f"\n==================================================")
    print(f"  {description}")
    print(f"==================================================")
    env_cmd = f"set PYTHONPATH=. && {sys.executable} {command}" if sys.platform == 'win32' else f"PYTHONPATH=. {sys.executable} {command}"
    result = os.system(env_cmd)
    if result != 0:
        print(f"\n[ERROR] fallo ejecutando: {command}")
        sys.exit(1)

def main():
    print(">>> INICIANDO PIPELINE COMPLETO Y BENCHMARKING LSTM-ECU...")
    
    # Step 0: Generar Jupyter Notebooks limpios
    run_step("Paso 0: Generando Jupyter Notebooks", "src/utils/generate_notebook.py")
    
    # Step 1: Preprocesamiento e Ingesta de Datos
    run_step("Paso 1: Procesamiento de Microdatos ENEMDU", "src/preprocessing/build_dataset.py")
    run_step("Paso 2: Descargando precios de Petroleo WTI", "src/preprocessing/download_wti.py")
    run_step("Paso 3: Fusionando datasets (INEC + IPC + WTI)", "src/preprocessing/merge_datasets.py")
    run_step("Paso 4: Estacionarizacion de Exogenas y Tensores 3D (Lookback T=12)", "src/preprocessing/build_features.py")
    
    # Step 2: Auditoria
    run_step("Paso 5: Auditoria Econometrica ADF", "src/models/01_data_audit.py")
    
    # Step 3: Entrenamiento de Modelos y Benchmarking DL
    run_step("Paso 6: Entrenando Red LSTM Principal (PyTorch)", "src/models/02_train_lstm.py")
    run_step("Paso 7: Benchmarking Arquitecturas DL (LSTM, GRU, BiLSTM, Attention)", "src/models/02b_train_dl_models.py")
    run_step("Paso 8: Entrenando Modelos Linea Base (XGBoost, VAR, SARIMA)", "src/models/03_train_baselines.py")
    
    # Step 4: Explicabilidad e Inferencia
    run_step("Paso 9: Analisis de Atribucion SHAP (XAI)", "src/models/04_shap_analysis.py")
    run_step("Paso 10: Test de Diebold-Mariano", "src/models/05_diebold_mariano.py")
    
    # Step 5: Experimentos de Simulación Contrafáctica (Stress Testing 117 Exps)
    run_step("Paso 11: Simulacion de 117 Experimentos Contrafacticos y Mapa de Calor 2D", "src/models/07_exogenous_counterfactual_experiments.py")
    run_step("Paso 12: Generacion de Graficos de Evaluacion Final", "src/visualization/01_plot_predictions.py")
    
    print("\n==================================================")
    print("SUCCESS: PIPELINE INTEGRAL EJECUTADO CON EXITO")
    print("==================================================\n")

if __name__ == "__main__":
    main()
