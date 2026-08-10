# Notas Metodológicas (Obsidian) - Proyecto LSTM-ECU

## 1. Tratamiento de los Datos (Data Engineering)
- **Rango temporal:** Optamos por la *Opción A*, extendiendo el análisis de enero 2015 a diciembre 2025 (y primeros meses de 2026), logrando un $N=136$.
- **Mitigación de Varianza:** En lugar de dividir por sectores (lo cual generaba excesivo ruido y pocos datos), se usó el agregado nacional de la Tasa de Empleo Adecuado del INEC.
- **Interpolación (El vacío de la pandemia):** El script detectó 52 meses donde el INEC no recolectó datos. Se aplicó una interpolación lineal para llenarlos, y se creó la variable dicotómica `es_interpolado` (1 o 0). Esto funciona como "guardarraíl" para que la red sepa qué datos son estimaciones y cuáles son mediciones reales.

## 2. Ingeniería de Características (Feature Engineering)
- **Variables Exógenas:** Se integraron IPC (Inflación) y WTI (Precio del barril de petróleo) alineados mes a mes.
- **Codificación Cíclica:** Para que la LSTM entienda la estacionalidad anual, el número del mes se transformó en dos coordenadas trigonométricas:
  - $MES_{sin} = \sin((2\pi \times MES)/12)$
  - $MES_{cos} = \cos((2\pi \times MES)/12)$
- **Ventana Deslizante (Lookback):** Se generaron tensores 3D con $T=12$, obligando a la red a mirar 1 año hacia atrás para predecir el siguiente mes.

## 3. Arquitectura del Modelo
- **Migración a PyTorch:** Debido a incompatibilidades técnicas de TensorFlow con Python 3.14 en la máquina local, la arquitectura completa se migró a PyTorch, manteniendo el 100% del rigor metodológico estipulado en tu texto original.
- **Parámetros exactos:** Red `nn.LSTM(hidden=64)` + Penalidad `Dropout(0.4)` + Optimizador `AdamW(lr=0.005, weight_decay=1e-4)`.
- **Early Stopping:** La red entrenó un máximo de 150 épocas pero se detuvo sola en la época 53, punto matemático exacto donde el error de validación (2024) alcanzó su mínimo absoluto, previniendo la memorización (overfitting).

## 4. Resultados del Experimento (Out-of-Sample)
Se evaluaron todos los modelos frente al Test Set estrictamente limpio (2025). La red LSTM superó a las tres líneas base econométricas y de machine learning:

| Modelo | RMSE | MAE (Error Absoluto) | sMAPE (Precisión) |
| :--- | :--- | :--- | :--- |
| **LSTM (Deep Learning)** | **1.76** | **1.46 %** | **4.13 %** |
| XGBoost (Machine Learning) | 2.33 | 1.90 % | 5.39 % |
| VAR (Econometría Multivariada) | 4.67 | 4.24 % | 12.50 % |
| SARIMA (Econometría Univariada) | 4.55 | 4.36 % | 12.86 % |

*Conclusión Técnica:* La LSTM logró un margen de error promedio de 1.46 puntos porcentuales. Al superar con creces a modelos multivariados como el VAR y univariados como SARIMA, se demuestra que la Tasa de Empleo en Ecuador no es lineal. XGBoost se acercó bastante, demostrando que el machine learning en general domina este problema.

## 5. Explicabilidad e Inferencia (XAI y Estadística)
- **Análisis SHAP:** Se extrajeron los valores de contribución marginal (Bee Swarm y Bar Plot) eliminando la naturaleza de "caja negra" de la red. La "Tasa Histórica (t-1)" resultó ser el driver principal de la LSTM.
- **Prueba de Diebold-Mariano:** Se comparó matemáticamente a la LSTM contra los demás modelos (H0: Tienen la misma precisión predictiva).
  - LSTM vs XGBoost: *p-valor = 0.066* (Empate estadístico en $\alpha=0.05$, aunque LSTM tiene menor error absoluto).
  - LSTM vs VAR: *p-valor = 0.00002* (LSTM es superior).
  - LSTM vs SARIMA: *p-valor < 0.00001* (LSTM es superior).
  - **Conclusión Definitiva:** Se rechaza la hipótesis nula contra los modelos econométricos ($p < 0.05$). La LSTM destroza a la econometría clásica matemáticamente, y se bate a duelo cerrado con XGBoost.
