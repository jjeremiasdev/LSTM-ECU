"""
Auditoría Científica de Datos y Propiedades Econométricas (LSTM-ECU)
Diagnóstico de la calidad de los datos y pruebas de estacionariedad.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from sklearn.ensemble import IsolationForest
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

def audit():
    filepath = 'data/raw/dataset.csv'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo {filepath} no existe.")

    df = pd.read_csv(filepath, parse_dates=['fecha'])
    df = df.set_index('fecha').sort_index()

    print("Valores Faltantes por Columna:")
    print(df.isnull().sum())
    print("\nFilas Duplicadas:", df.duplicated().sum())

    target = 'empleo_adecuado'
    print("\n--- Augmented Dickey-Fuller (ADF) ---")
    adf = adfuller(df[target].dropna())
    print(f"ADF Statistic: {adf[0]:.4f}")
    print(f"p-value: {adf[1]:.4f}")

    print("\n--- KPSS Test ---")
    kpss_stat = kpss(df[target].dropna(), regression='c', nlags="auto")
    print(f"KPSS Statistic: {kpss_stat[0]:.4f}")
    print(f"p-value: {kpss_stat[1]:.4f}")

    print("\n--- Phillips-Perron (PP) Test ---")
    pp = PhillipsPerron(df[target].dropna())
    print(f"PP Statistic: {pp.stat:.4f}")
    print(f"p-value: {pp.pvalue:.4f}")

if __name__ == "__main__":
    audit()
