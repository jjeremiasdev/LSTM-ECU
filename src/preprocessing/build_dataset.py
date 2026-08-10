import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import glob
import os
import gc

def procesar_enemdu(input_folder, output_file):
    print("Iniciando procesamiento de microdatos ENEMDU...")
    archivos_csv = glob.glob(os.path.join(input_folder, "*.csv"))
    # Si usas SPSS descomenta la siguiente línea y la importación de pyreadstat
    # archivos_sav = glob.glob(os.path.join(input_folder, "*.sav"))
    # import pyreadstat
    
    resultados = []

    for archivo in archivos_csv:
        print(f"Procesando: {os.path.basename(archivo)}")
        try:
            # chunksize si la memoria es limitada, pero read_csv usualmente soporta un archivo a la vez
            df = pd.read_csv(archivo, low_memory=False, sep=';') # Ajusta el separador si es coma
            
            # 1. Identificar columnas temporales
            # ENEMDU suele usar 'anio' y 'mes', o 'periodo'
            anio_col = 'anio' if 'anio' in df.columns else 'p04' # Ejemplo, debes ajustar el nombre exacto
            mes_col = 'mes' if 'mes' in df.columns else 'periodo'
            
            if anio_col not in df.columns or mes_col not in df.columns:
                print(f"  Advertencia: No se encontraron columnas de fecha en {archivo}. Ignorando.")
                continue
                
            # Tomar el primer valor como representativo del archivo
            anio_val = df[anio_col].dropna().iloc[0]
            mes_val = df[mes_col].dropna().iloc[0]
            
            # Limpieza básica de la fecha
            try:
                # Mapear trimestre a mes si es necesario (ej: Trimestre 1 = Marzo -> mes 3)
                fecha = pd.to_datetime(f"{int(anio_val)}-{int(mes_val):02d}-01")
            except Exception as e:
                print(f"  Error convirtiendo fecha ({anio_val}-{mes_val}): {e}")
                continue

            # 2. Convertir nombres de columnas a minúsculas para uniformidad
            df.columns = [str(c).lower().strip() for c in df.columns]

            # 3. Nombres de columnas según diccionario (Ajustar si cambian con los años)
            # condact: Condición de Actividad (1=Adecuado, etc. Revisa tu diccionario de SPSS para el código exacto)
            # fexp: Factor de expansión
            # peaa: Población Económicamente Activa (usualmente >= 15 años y activos)
            
            col_condact = 'condact' if 'condact' in df.columns else 'empleo'
            col_fexp = 'fexp'
            col_peaa = 'peaa'

            if col_fexp not in df.columns:
                print(f"  Error: No se encontró 'fexp' en {archivo}")
                continue
                
            # Asegurar numérico
            df[col_fexp] = pd.to_numeric(df[col_fexp], errors='coerce').fillna(0)

            # 4. Cálculo de Empleo Adecuado
            # ADVERTENCIA: Debes reemplazar 'CódigoEmpleoAdecuado' por el número/texto exacto del INEC
            # Ejemplo: 1 = Empleo Adecuado
            CODIGO_ADECUADO = 1 
            
            if col_condact in df.columns:
                # Filtrar PEA (si existe una columna PEAA que marca 1 para activos)
                if col_peaa in df.columns:
                    pea_df = df[df[col_peaa] == 1]
                else:
                    # Si no hay bandera PEAA, asumimos que PEA son los que tienen alguna condición de actividad (diferente de inactivos)
                    pea_df = df.dropna(subset=[col_condact])
                
                total_pea = pea_df[col_fexp].sum()
                
                # Filtrar Empleo Adecuado
                empleo_adecuado_df = pea_df[pea_df[col_condact] == CODIGO_ADECUADO]
                total_adecuado = empleo_adecuado_df[col_fexp].sum()
                
                tasa = (total_adecuado / total_pea) * 100 if total_pea > 0 else np.nan
                
                resultados.append({
                    'fecha': fecha,
                    'tasa_empleo_adecuado': tasa
                })
            else:
                print(f"  Error: No se encontró la columna {col_condact} en {archivo}")

            # Liberar memoria
            del df
            gc.collect()

        except Exception as e:
            print(f"  Error crítico procesando {archivo}: {e}")

    if not resultados:
        print("No se extrajeron datos.")
        return

    # 5. Consolidar, ordenar y guardar
    df_final = pd.DataFrame(resultados)
    df_final = df_final.sort_values('fecha').drop_duplicates('fecha')
    
    # 6. (Opcional) Unir IPC y WTI si tienes un CSV externo ya preparado
    # df_externo = pd.read_csv("ruta_ipc_wti.csv", parse_dates=['fecha'])
    # df_final = pd.merge(df_final, df_externo, on='fecha', how='left')

    # Añadimos la columna booleana para interpolación (los vacíos se tratarán en el notebook de auditoría)
    df_final['es_interpolado'] = 0

    df_final.to_csv(output_file, index=False)
    print(f"\\nProceso exitoso. Dataset consolidado guardado en: {output_file}")
    print(df_final.head())

if __name__ == '__main__':
    # RUTAS RELATIVAS (Ejecutar este script desde la raíz del proyecto LSTM-ECU)
    INPUT_FOLDER = 'data/raw/microdatos'
    OUTPUT_FILE = 'data/raw/dataset.csv'
    
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"He creado la carpeta '{INPUT_FOLDER}'. Por favor, coloca ahí tus archivos .csv de la ENEMDU.")
    else:
        procesar_enemdu(INPUT_FOLDER, OUTPUT_FILE)
