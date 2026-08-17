import json
import os

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\\n' for line in text.split('\\n')][:-1]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\\n' for line in text.split('\\n')][:-1]
    })

text_intro = """# FASE 1: Auditoría Científica de Datos
Este notebook realiza un diagnóstico exhaustivo de la calidad de los datos y las propiedades econométricas de la serie temporal (Tasa de Empleo Adecuado, IPC, WTI).
El objetivo es responder: ¿Qué puede aprender realmente la LSTM de esta serie? ¿Es suficiente N=60?

Asegúrese de colocar el archivo de datos consolidado en `../data/raw/dataset.csv` antes de ejecutar."""
add_markdown(text_intro)

code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import ruptures as rpt

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
%matplotlib inline"""
add_code(code_imports)

text_load = """## 1. Carga de Datos y Calidad Temporal"""
add_markdown(text_load)

code_load = """# Intentar cargar el archivo. Si no existe, se generará una advertencia y el notebook se detendrá aquí hasta que se provean los datos.
import os

filepath = '../data/raw/dataset.csv'
if not os.path.exists(filepath):
    raise FileNotFoundError(f"El archivo {filepath} no existe. Por favor, agregue el dataset consolidado.")

df = pd.read_csv(filepath, parse_dates=['fecha'])
df = df.set_index('fecha')
df = df.sort_index()
display(df.head())"""
add_code(code_load)

text_calidad = """## 2. Calidad de Datos (Valores Faltantes y Duplicados)"""
add_markdown(text_calidad)

code_calidad = """print("Valores Faltantes por Columna:")
print(df.isnull().sum())
print("\\nFilas Duplicadas:", df.duplicated().sum())
print("\\nConsistencia Cronológica (Frecuencia Mensual):")
inferred_freq = pd.infer_freq(df.index)
print(f"Frecuencia inferida: {inferred_freq}")

display(df.describe())"""
add_code(code_calidad)

text_outliers = """## 3. Valores Extremos (Outliers)
Aplicamos IQR, Z-Score e Isolation Forest solo como diagnóstico. No se eliminan los outliers, se evalúa su justificación económica (e.g., Shock COVID-19)."""
add_markdown(text_outliers)

code_outliers = """def detect_outliers(series):
    # IQR
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))]
    
    # Z-Score
    z_scores = (series - series.mean()) / series.std()
    outliers_z = series[np.abs(z_scores) > 3]
    
    return outliers_iqr.index, outliers_z.index

target = 'empleo_adecuado' # Ajustar si el nombre de columna es diferente
iqr_idx, z_idx = detect_outliers(df[target].dropna())

# Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
iso_preds = iso.fit_predict(df[[target]].dropna())
iso_outliers = df[[target]].dropna()[iso_preds == -1]

plt.figure(figsize=(15,5))
plt.plot(df.index, df[target], label='Original', color='blue')
plt.scatter(iso_outliers.index, iso_outliers[target], color='red', label='Outliers (Isolation Forest)')
plt.title(f'Detección de Valores Extremos: {target}')
plt.legend()
plt.show()

print(f"Outliers detectados por IQR: {len(iqr_idx)}")
print(f"Outliers detectados por Z-Score (abs > 3): {len(z_idx)}")
print(f"Outliers detectados por Isolation Forest: {len(iso_outliers)}")"""
add_code(code_outliers)

text_trend = """## 4. Tendencia y Estacionalidad (Descomposición STL)"""
add_markdown(text_trend)

code_trend = """stl = STL(df[target].dropna(), period=12)
res = stl.fit()
fig = res.plot()
fig.set_size_inches(15, 8)
plt.show()"""
add_code(code_trend)

text_memory = """## 5. Memoria Temporal (ACF y PACF)
Determina la correlación de la serie con sus rezagos. Fundamental para justificar un lookback de 12 meses."""
add_markdown(text_memory)

code_memory = """fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(df[target].dropna(), lags=24, ax=axes[0], title='Autocorrelación (ACF)')
plot_pacf(df[target].dropna(), lags=24, ax=axes[1], title='Autocorrelación Parcial (PACF)')
plt.show()"""
add_code(code_memory)

text_stationarity = """## 6. Pruebas de Estacionariedad
Pruebas formales para determinar si la serie requiere diferenciación."""
add_markdown(text_stationarity)

code_stationarity = """def test_stationarity(series):
    series = series.dropna()
    print("--- Augmented Dickey-Fuller (ADF) ---")
    adf = adfuller(series)
    print(f"ADF Statistic: {adf[0]:.4f}")
    print(f"p-value: {adf[1]:.4f}")
    
    print("\\n--- KPSS Test ---")
    kpss_stat = kpss(series, regression='c', nlags="auto")
    print(f"KPSS Statistic: {kpss_stat[0]:.4f}")
    print(f"p-value: {kpss_stat[1]:.4f}")
    
    print("\\n--- Phillips-Perron (PP) Test ---")
    pp = PhillipsPerron(series)
    print(f"PP Statistic: {pp.stat:.4f}")
    print(f"p-value: {pp.pvalue:.4f}")

test_stationarity(df[target])"""
add_code(code_stationarity)

text_breaks = """## 7. Rupturas Estructurales (Cambios de Régimen)"""
add_markdown(text_breaks)

code_breaks = """series_np = df[target].dropna().values
algo = rpt.Pelt(model="rbf").fit(series_np)
result = algo.predict(pen=10)

rpt.display(series_np, result, figsize=(15, 5))
plt.title('Detección de Rupturas Estructurales (PELT)')
plt.show()"""
add_code(code_breaks)

text_corr = """## 8. Correlaciones y Multicolinealidad"""
add_markdown(text_corr)

code_corr = """corr_pearson = df.corr(method='pearson')
corr_spearman = df.corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title('Correlación de Pearson')
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title('Correlación de Spearman')
plt.show()

# VIF
X = df.dropna().drop(columns=[target], errors='ignore')
if not X.empty and X.select_dtypes(include=np.number).shape[1] > 1:
    X_num = X.select_dtypes(include=np.number)
    X_vif = X_num.assign(const=1)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
    display(vif_data)"""
add_code(code_corr)

text_scaling = """## 9. Análisis de Escalamiento
Comparación de las distribuciones tras aplicar distintos escaladores."""
add_markdown(text_scaling)

code_scaling = """if not df.empty:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    sns.kdeplot(df[target].dropna(), ax=axes[0], fill=True)
    axes[0].set_title('Original Distribution')
    
    ss = StandardScaler()
    scaled_ss = ss.fit_transform(df[[target]].dropna())
    sns.kdeplot(scaled_ss.flatten(), ax=axes[1], fill=True)
    axes[1].set_title('StandardScaler')
    
    rs = RobustScaler()
    scaled_rs = rs.fit_transform(df[[target]].dropna())
    sns.kdeplot(scaled_rs.flatten(), ax=axes[2], fill=True)
    axes[2].set_title('RobustScaler')
    
    mms = MinMaxScaler()
    scaled_mms = mms.fit_transform(df[[target]].dropna())
    sns.kdeplot(scaled_mms.flatten(), ax=axes[3], fill=True)
    axes[3].set_title('MinMaxScaler')
    
    plt.tight_layout()
    plt.show()"""
add_code(code_scaling)

text_conclusion = """## 10. Conclusiones Metodológicas
En base a los resultados obtenidos:
1. ¿Es N=60 suficiente o se observa inestabilidad estadística severa?
2. ¿Existen cambios de régimen (ej. COVID) que obligan a usar variables dummy?
3. ¿Cuál escalador preserva mejor la forma original de los datos (sin comprimir la varianza artificialmente)?
4. ¿El ACF confirma que un lookback de 12 meses es óptimo para extraer la señal?

**(Responder después de observar los resultados empíricos)**"""
add_markdown(text_conclusion)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs('notebooks', exist_ok=True)
output_nb_path = 'notebooks/01_Data_Audit.ipynb'
with open(output_nb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print(f"Notebook generado exitosamente en {output_nb_path}")
