# Metodología (Texto Revisado y Adaptado)

*Nota: Aquí tienes tu propio texto reescrito para reflejar exactamente lo que hicimos (Opción A). Puedes copiar y pegar esto en tu documento oficial.*

---

**Datos, variables y periodo de estudio**
La investigación permitió el análisis a un periodo estratégico de 132 meses (N=132), comprendido entre enero de 2015 y diciembre de 2025. La variable endógena u objetivo $Y_t$, definida como la tasa general de empleo adecuado a nivel nacional, se adoptó directamente de los tabulados consolidados oficiales del Instituto Nacional de Estadística y Censos (INEC, 2026). Al utilizar el agregado macroeconómico en lugar de microdatos sectorizados, el experimento mitiga el ruido estadístico de muestras pequeñas (varianza) y asegura la consistencia de la serie a lo largo de 11 años de cambios demográficos.

El horizonte temporal abarca tres regímenes económicos diferenciados:
1. **Fase de Desgaste Inercial (2015 – 2019):** Por su parte, el periodo 2015 – 2019 se conceptualiza como la etapa de desgaste inercial y desacelerado. Tras el 2014, el mercado laboral ecuatoriano se vio afectado por su economía dolarizada al registrar un deterioro progresivo, carente de la versatilidad de la moneda. Los ingresos petroleros fueron en declive y esto se evidenció en el sector laboral, generando subempleo y sobre todo informalidad.
2. **Shock Estructural Exógeno (2020):** Esta dinámica de desgaste significó un 2020 crítico ante una crisis sanitaria que representó el shock de mayor impacto para la economía. La emergencia sanitaria global por la COVID-19, sumada a la parálisis operativa del sector productivo y al colapso en la cotización del crudo, provocó que se desplomaran sin precedentes los niveles de ocupación formal y obligó a la suspensión temporal de la encuesta ENEMDU por parte del INEC. Estadísticamente, el año 2020 operó como un punto de inflexión y ruptura de régimen —conocido como *structural break* por su término en inglés—, descartando la linealidad y la estacionalidad estricta por la que se rigen los modelos econométricos, justificando así la necesidad de algoritmos con capacidad de memoria a corto y largo plazo como lo son las redes LSTM.
3. **Periodo de Quiebre y Reincorporación Asimétrica (2021 – 2025):** Durante los periodos 2021 – 2025 las fluctuaciones se han agudizado a tal punto que estas irregularidades forman parte del periodo de reincorporación asimétrica de la constante del trabajo. Periodos como el postpandemia enmarcaron un antes y un después cuando de mercado laboral se trata; asimismo, la crisis de seguridad y la inestabilidad sociopolítica son un desafío. Este periodo se lo puede denominar como el periodo de quiebre, el cual se traduce estadísticamente a relaciones no lineales y alta volatilidad cuando de una serie de tiempo laboral se trata.

		Para manejar la falta de recolección de encuestas ENEMDU durante los 52 meses de disrupción y suspensión, se aplicó una interpolación matemática lineal continua. Para evitar sesgos algorítmicos sobre estos puntos interpolados, se generó una variable exógena dicotómica o de control ($V_c = 1$ si el dato fue interpolado, $0$ caso contrario), lo que le permite a la red neuronal comprender qué meses son estimaciones y cuáles son recolecciones de campo reales.

**Agregación temporal y mitigación de ruido**
Según Nekarda & Barnichon (2012), el pronóstico de variables agregadas a nivel nacional puede encubrir dinámicas subyacentes, pero para modelos de *Deep Learning* aplicados a la macroeconomía, la estabilidad de la varianza es preferible ante el ruido de la sub-sectorización extrema. El uso de $N=132$ permite generar una serie homogénea y suficientemente amplia para el aprendizaje profundo de una red recurrente, evadiendo la maldición de la dimensionalidad que ocurriría al segmentar por sectores.

**Características temporales y exógenas**
Los mercados laborales absorben con facilidad los shocks externos. En economías dolarizadas e inelásticas como la de Ecuador (Orellana Baraja et al., 2025), el empleo responde a dinámicas globales y monetarias. Por este contexto, el modelo integra dos variables exógenas fundamentales: la inflación (Índice de Precios al Consumidor bruto - IPC) y el precio internacional del barril de petróleo WTI. 

Además, para otorgarle al algoritmo la capacidad de comprender el calendario gregoriano (ciclos anuales), se transformó la linealidad del tiempo mediante codificación cíclica geométrica (Zheng & Casari, 2018). Se reemplazó el número del mes por sus componentes vectoriales, resolviendo así la distancia falsa entre diciembre y enero:
$MES_{sin} = \sin((2\pi \times MES)/12)$
$MES_{cos} = \cos((2\pi \times MES)/12)$

**Secuencia dimensional**
A diferencia de los modelos econométricos tradicionales que procesan vectores unidimensionales, las arquitecturas recurrentes requieren un tensor tridimensional (Bontempi et al., 2013). Para ello, se aplicó el método de ventana deslizante (*lookback*) estableciendo un $T=12$ (12 meses históricos) para predecir el horizonte $T+1$. Athanasopoulos & Hyndman (2018) validan que un $T=12$ es óptimo para capturar efectos de correlación cíclica anual.

**Arquitectura del core predictivo de la red LSTM**
El modelo propuesto es una red neuronal recurrente LSTM (Hochreiter & Schmidhuber, 1997) que utiliza compuertas de olvido, entrada y salida para separar el ruido transitorio de los patrones econométricos estructurales, resolviendo el problema del desvanecimiento del gradiente. Para mitigar la memorización espuria (overfitting), la arquitectura incorpora una capa de Dropout con una penalidad de 0.4 (Srivastava et al., 2014) y optimiza sus hiperparámetros mediante ADAMW, el cual desacopla eficazmente la decadencia de pesos en ambientes complejos (Loshchilov & Hutter, 2019).

**Modelos significativos (Líneas base)**
Para demostrar la superioridad o igualdad de la LSTM, se contrastó su rendimiento contra tres marcos epistemológicos:
1. **SARIMA:** Modelo lineal univariado que asume estacionalidad estricta sin contemplar variables exógenas.
2. **VAR (Vectores Autorregresivos):** Modelo multivariado de ecuaciones simultáneas que asume relaciones de proporcionalidad constante y lineal (Sims, 1980) entre el empleo, el IPC y el WTI.
3. **XGBoost:** Algoritmo de ensamble arbóreo que mapea alta no-linealidad, pero que procesa el espacio de manera tabular (estática), careciendo del mecanismo interno para comprender el orden secuencial cronológico continuo (Chen & Guestrin, 2016).

**Estrategia de evaluación y métricas**
Para evitar la alteración de la cronología, se rechazaron técnicas como K-Fold Cross-Validation, optando por una validación de origen móvil (Tashman, 2000; Cerqueira et al., 2020) dividida en:
- **Conjunto de Entrenamiento:** Enero 2015 a Diciembre 2023. Permite a la red calibrar pesos asimilando diversos ciclos económicos y el fuerte shock inicial de la pandemia.
- **Conjunto de Validación:** Enero 2024 a Diciembre 2024 (12 meses). Utilizado estrictamente para la optimización de hiperparámetros (como la tasa de aprendizaje) sin contaminar los datos de prueba.
- **Conjunto de Prueba:** Enero 2025 a Diciembre 2025 (12 meses). Segmento aislado de comprobación final (Out-of-sample).

La cuantificación del error se evaluó mediante tres métricas (Makridakis et al., 2018): **RMSE** para penalizar desviaciones severas que acarrearían costos socioeconómicos altos; **MAE** para expresar la magnitud promedio del error pronosticado; y **sMAPE** para corregir la asimetría del MAPE y permitir una evaluación porcentual estandarizada.

**Inferencia estadística y atribución de la XAI**
Para garantizar que la reducción del error no sea producto del azar frente a una varianza finita, se implementó la prueba de Diebold-Mariano con $\alpha = 0.05$ (Diebold & Mariano, 1995). Finalmente, se utilizó el algoritmo de interpretabilidad post-hoc SHAP (Lundberg & Lee, 2017) para aislar la contribución marginal de las variables exógenas (IPC, WTI) en las decisiones no lineales de la red, abriendo la "caja negra" a la econometría.
