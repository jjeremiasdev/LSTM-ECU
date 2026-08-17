import pandas as pd
import numpy as np
import glob
import os
import gc

def procesar_enemdu(input_folder, output_file):
    print("Iniciando procesamiento de microdatos ENEMDU...")
    archivos_csv = glob.glob(os.path.join(input_folder, "*.csv"))
    archivos_sav = glob.glob(os.path.join(input_folder, "*.sav"))
    todos_archivos = archivos_csv + archivos_sav

    if not todos_archivos:
        print(f"No se encontraron archivos .csv o .sav en '{input_folder}'.")
        if os.path.exists('data/raw/tasas_inec.csv'):
            print("Utilizando la serie consolidada de tasas oficiales de 'data/raw/tasas_inec.csv'.")
        return

    resultados = []

    for archivo in todos_archivos:
        print(f"Procesando microdatos: {os.path.basename(archivo)}")
        try:
            if archivo.endswith('.sav'):
                try:
                    import pyreadstat
                    df, _ = pyreadstat.read_sav(archivo)
                except ImportError:
                    print(f"  Advertencia: 'pyreadstat' no instalado. Omitiendo {archivo}.")
                    continue
            else:
                df = pd.read_csv(archivo, low_memory=False, sep=';')

            df.columns = [str(c).lower().strip() for c in df.columns]

            anio_col = next((c for c in ['anio', 'año', 'p04', 'periodo'] if c in df.columns), None)
            mes_col = next((c for c in ['mes', 'periodo', 'trimestre'] if c in df.columns), None)

            if not anio_col or not mes_col:
                print(f"  Advertencia: No se detectaron columnas temporales en {archivo}.")
                continue

            anio_val = df[anio_col].dropna().iloc[0]
            mes_val = df[mes_col].dropna().iloc[0]

            try:
                fecha = pd.to_datetime(f"{int(float(anio_val))}-{int(float(mes_val)):02d}-01")
            except Exception as e:
                print(f"  Error parseando fecha ({anio_val}-{mes_val}): {e}")
                continue

            col_condact = next((c for c in ['condact', 'empleo', 'p10a'] if c in df.columns), None)
            col_fexp = next((c for c in ['fexp', 'factor', 'peso'] if c in df.columns), None)
            col_peaa = next((c for c in ['peaa', 'pea'] if c in df.columns), None)

            if not col_fexp or not col_condact:
                print(f"  Error: Faltan columnas 'fexp' o 'condact' en {archivo}.")
                continue

            df[col_fexp] = pd.to_numeric(df[col_fexp], errors='coerce').fillna(0)
            df[col_condact] = pd.to_numeric(df[col_condact], errors='coerce')

            # Condición 1 = Empleo Adecuado en codificación INEC
            pea_df = df[df[col_peaa] == 1] if col_peaa and col_peaa in df.columns else df.dropna(subset=[col_condact])
            total_pea = pea_df[col_fexp].sum()
            total_adecuado = pea_df[pea_df[col_condact] == 1][col_fexp].sum()

            tasa = (total_adecuado / total_pea) * 100 if total_pea > 0 else np.nan

            resultados.append({'fecha': fecha, 'empleo_adecuado': tasa})

            del df
            gc.collect()

        except Exception as e:
            print(f"  Error procesando {archivo}: {e}")

    if resultados:
        df_final = pd.DataFrame(resultados).sort_values('fecha').drop_duplicates('fecha')
        df_final['es_interpolado'] = 0
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_final.to_csv(output_file, index=False)
        print(f"\nProceso exitoso. Microdatos consolidados en: {output_file}")

if __name__ == '__main__':
    INPUT_FOLDER = 'data/raw/microdatos'
    OUTPUT_FILE = 'data/raw/dataset_microdatos.csv'
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    procesar_enemdu(INPUT_FOLDER, OUTPUT_FILE)
