# Notas Metodológicas y Justificaciones Técnicas (Documentación Obsidian - Proyecto LSTM-ECU)

Este documento contiene la justificación metodológica, econométrica y computacional de cada decisión tomada en la construcción, optimización y auditoría del modelo de Deep Learning para el pronóstico de la Tasa de Empleo Adecuado en Ecuador.

---

## 1. Justificación de la Arquitectura y Selección del Modelo Principal (LSTM vs GRU)

### A. ¿Por qué la LSTM es el Modelo y Solución Principal del Proyecto?
- **Objetivo Científico:** La pregunta de investigación busca responder si una red neuronal con memoria de largo plazo (**LSTM**) es capaz de superar la rigidez y las presunciones de linealidad de los modelos econométricos tradicionales (**SARIMA** y **VAR**).
- **Resultado Demostrado:** La LSTM reduce el error cuadrático medio de **4.55% (SARIMA)** y **4.67% (VAR)** a solo **1.57% (RMSE)** y **1.33% (MAE)**. La prueba de Diebold-Mariano confirma que esta diferencia es estadísticamente significativa ($p < 0.0001$). Por lo tanto, la LSTM cumple al 100% el objetivo principal y se sostiene como la propuesta central del estudio.

### B. ¿Cómo encuadrar la GRU en el Artículo Científico?
- **Encuadre Metodológico (Análisis de Sensibilidad de Arquitectura):** En artículos científicos arbitrados, evaluar la **GRU** junto a la LSTM no debilita la propuesta, sino que la fortalece como un **Análisis de Ablación/Sensibilidad**.
- **Justificación Teórica de la GRU (Ley de Parsimonia):** La LSTM utiliza 3 compuertas (Olvido $f_t$, Entrada $i_t$, Salida $o_t$) y un estado de celda $c_t$, acumulando más parámetros a calibrar. La GRU simplifica la unidad a 2 compuertas (Actualización $z_t$ y Reinicio $r_t$). En muestras macroeconómicas de tamaño moderado ($N=132$), la estructura más parsimoniosa de la GRU reduce la varianza del estimador y previene el sobreajuste (*overfitting*), alcanzando un RMSE de **1.43%**.
- **Conclusión para la Redacción:** Se presenta la **LSTM como la solución de Deep Learning de referencia** y se concluye que las arquitecturas recurrentes compactas (GRU/LSTM) dominan la no-linealidad del empleo ecuatoriano.

---

## 2. Tratamiento de Datos e Ingeniería de Características (Data Engineering)

### A. Ampliación del Tamaño de Muestra ($N=132$ meses, 2015–2025)
- **Justificación:** Inicialmente se planteó usar 60 meses (2021–2025). Sin embargo, para entrenar redes neuronales profundas sin caer en memorización espuria, $N=60$ resultaba insuficiente. Ampliar el horizonte a 11 años ($N=132$) permitió capturar tres regímenes económicos: la tendencia pre-pandemia (2015–2019), el shock estructural COVID-19 (2020) y la reactivación asimétrica post-pandemia (2021–2025).

### B. Tratamiento del Vacío de la Pandemia e Interpolación (`es_interpolado` / $V_c$)
- **Justificación:** Durante 52 meses no se recolectó la encuesta ENEMDU de forma presencial continua. Para evitar saltos cronológicos que romperían el tensor 3D de la LSTM, se aplicó una interpolación matemática lineal.
- **Variable Guardarraíl ($V_c$):** Se creó una variable dicotómica exógena (`es_interpolado` $\in \{0, 1\}$). Esto le indica explícitamente a la red qué datos son mediciones reales de campo del INEC y cuáles son estimaciones matemáticas, previniendo que la red aprenda relaciones falsas del algoritmo de interpolación.
- **Preservación en Simulación:** En el conjunto de test (2025) y durante los 117 experimentos de simulación, $V_c = 0$ de forma constante, ya que se evalúa la respuesta del modelo bajo condiciones de medición real e íntegra en campo.

### C. Estacionarización de Variables Exógenas ($\Delta \% \text{IPC}_t$ y $\Delta \% \text{WTI}_t$)
- **Justificación Econométrica:** El Índice de Precios al Consumidor (IPC) en nivel bruto es una serie no estacionaria con tendencia determinística creciente (ADF $p = 0.3015$). Si se ingresa el IPC en nivel bruto a un `MinMaxScaler` ajustado en el periodo 2015–2023, los datos de test de 2025 escapan de la escala $[0, 1]$ aprendida (*out-of-bounds scaling*).
- **Transformación:** Se calcularon las tasas de variación porcentual mensual:
  $$\Delta \% \text{IPC}_t = \frac{\text{IPC}_t - \text{IPC}_{t-1}}{\text{IPC}_{t-1}} \times 100, \quad \Delta \% \text{WTI}_t = \frac{\text{WTI}_t - \text{WTI}_{t-1}}{\text{WTI}_{t-1}} \times 100$$
- **Resultado:** Las exógenas se convirtieron en series estacionarias centradas cerca de cero. La red aprende la respuesta del empleo ante *shocks puntuales inflacionarios o petroleros*, reduciendo el RMSE global del experimento de **1.76 a 1.43**.

### D. Codificación Cíclica de Fechas ($\text{MES}_{\sin}, \text{MES}_{\cos}$)
- **Justificación:** Reemplazar el número del mes ($1$ a $12$) por coordenadas trigonométricas en el círculo unitario evita que la red asuma erróneamente que diciembre ($12$) y enero ($1$) están distanciados por 11 unidades, cuando en el tiempo real son meses consecutivos ($2\pi$ de periodicidad).
- **Preservación en Simulación:** En los 117 experimentos se preservan los valores reales de $\text{MES}_{\sin}$ y $\text{MES}_{\cos}$ de cada mes del año evaluado, ya que el calendario es determinístico y constante.

---

## 3. Calibración y Optimizadores en PyTorch

### A. Migración a PyTorch
- **Justificación:** Facilita el cálculo del gradiente estocástico y la compatibilidad con herramientas de explicabilidad como SHAP `GradientExplainer`.

### B. Optimizador AdamW ($lr=0.005, weight\_decay=1\times 10^{-4}$)
- **Justificación:** A diferencia de Adam estándar, **AdamW desacopla la caída de pesos (weight decay) del gradiente**, evitando que los pesos de las compuertas se hinchen artificialmente ante el ruido de la serie macroeconómica (Loshchilov & Hutter, 2019).

### C. Dropout (0.4) y Early Stopping (Patience = 20)
- **Justificación:** La capa de Dropout desconecta aleatoriamente el 40% de las neuronas en el entrenamiento, forzando a la red a no depender de una sola conexión. El Early Stopping monitorea la pérdida en el conjunto de validación limpia (2024), deteniendo el entrenamiento en la época 51 para garantizar la máxima capacidad de generalización en el conjunto de prueba (2025).

---

## 4. Inferencia Estadística y Explicabilidad (XAI)

### A. Prueba de Diebold-Mariano (Inferencia de Precisión Predictiva)
- **Justificación:** Para demostrar que el menor error de la LSTM y GRU no fue un evento fortuito de la muestra, la prueba de Diebold-Mariano evaluó la hipótesis nula $H_0$: *los modelos tienen la misma precisión*.
- **Resultados:** 
  - LSTM vs SARIMA: $DM = -6.39, p < 0.0001$ $\rightarrow$ Se rechaza $H_0$. LSTM es significativamente superior.
  - LSTM vs VAR: $DM = -4.16, p = 0.00003$ $\rightarrow$ Se rechaza $H_0$. LSTM es significativamente superior.
  - LSTM vs XGBoost: $DM = -1.46, p = 0.143$ $\rightarrow$ Empate técnico estadístico.

### B. Atribución SHAP (Explainable AI)
- **Justificación:** Elimina la objeción de "caja negra". SHAP cuantifica la contribución marginal de cada variable en el tensor 3D, demostrando que el rezago histórico ($t-1$) es el driver principal, mientras que la inflación ($\Delta\% \text{IPC}$) y el petróleo ($\Delta\% \text{WTI}$) actúan como modificadores no lineales.

---

## 5. Simulación Exhaustiva de 117 Experimentos Contrafácticos y Guía del Archivo Excel (`reports/matriz_experimentos_stress_testing.xlsx`)

### A. Metodología de Umbrales Económicos Realistas (Grilla 13x9)
- **Justificación:** Para evaluar la sensibilidad del modelo sin caer en escenarios irracionales (como inflaciones del 50%), se construyó una grilla acotada por los percentiles históricos reales de Ecuador (2015–2025):
  - **Petróleo WTI ($\Delta\% \text{WTI}$):** 13 niveles de shock desde $-30\%$ hasta $+30\%$ en pasos del $5\%$.
  - **Inflación ($\Delta\% \text{IPC}$):** 9 niveles de shock desde $-1.0\%$ hasta $+3.0\%$ en pasos del $0.5\%$.
  - Total: $13 \times 9 = 117$ combinaciones cruzadas.

### B. Diccionario de Columnas del Archivo Excel (`reports/matriz_experimentos_stress_testing.xlsx`)

#### Hoja 1: `Matriz_Pivote_2D`
- **Filas:** 9 niveles de shock en la tasa de inflación mensual ($\Delta \% \text{IPC}$).
- **Columnas:** 13 niveles de shock en la variación del precio del petróleo ($\Delta \% \text{WTI}$).
- **Celdas:** Tasa de Empleo Adecuado Promedio Estimada (%) para el periodo evaluado.

#### Hoja 2: `Detalle_117_Experimentos`
1. `Exp_ID`: Identificador correlativo único (`EXP_001` a `EXP_117`).
2. `Shock_WTI_%`: Variación porcentual del precio del petróleo WTI inyectada en la simulación.
3. `Shock_IPC_%`: Tasa de inflación mensual porcentual inyectada en la simulación.
4. `Empleo_Adecuado_Promedio_%`: **Resultado principal del modelo**; Tasa de Empleo Adecuado promedio predicha (%) bajo el escenario macroeconómico.
5. `Desviacion_Estandar`: Volatilidad o dispersión de la predicción entre los 12 meses evaluados.
6. `Empleo_Min_%`: Tasa mínima predicha en el mes más desfavorable bajo ese escenario.
7. `Empleo_Max_%`: Tasa máxima predicha en el mes más favorable bajo ese escenario.
8. `Escenario`: Categoría macroeconómica analítica: *"Estagflación"*, *"Bonanza"* o *"Simulación Estándar"*.

### C. Justificación Estadística y Papel de la Desviación Estándar ($\sigma$) en los Experimentos
- **¿Qué mide en la simulación?:** Mide la dispersión o variabilidad intrínseca de las 12 predicciones mensuales intra-anuales generadas para cada escenario.
- **¿Por qué es importante en la defensa de tesis/paper?:**
  1. **Indicador de Volatilidad Macroeconómica:** Revela si un shock no solo altera la tasa promedio de empleo, sino si lo vuelve más inestable o fluctuante a lo largo del año.
  2. **Análisis de Riesgo e Intervalos de Tolerancia:** Permite construir intervalos empíricos de variabilidad ($\text{Promedio} \pm 1\sigma$), otorgando al tomador de decisiones gubernamental una banda de fluctuación esperada para la planificación del mercado laboral.
