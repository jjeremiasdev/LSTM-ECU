import pandas as pd
import os

def mes_a_numero_corto(mes_str):
    meses = {
        'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'ago': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
    }
    m = str(mes_str)[:3].lower()
    return meses.get(m, '01')

def procesar_tasas_inec(filepath):
    print("Procesando tasas del INEC...")
    # Leer el CSV saltando la primera linea vacia (header=1)
    df = pd.read_csv(filepath, sep=';', encoding='latin1', header=1)
    
    # Limpiar nombres de columnas
    df.columns = [str(c).strip() for c in df.columns]
    
    # Buscar la columna de indicadores
    col_indicador = [c for c in df.columns if 'indicador' in c.lower()]
    if not col_indicador:
        raise ValueError("No se encontró la columna 'Indicadores' en el archivo del INEC.")
    col_indicador = col_indicador[0]
    
    # Filtrar Empleo Adecuado
    df_adecuado = df[df[col_indicador].str.contains("Empleo Adecuado", case=False, na=False)].copy()
    
    col_periodo = 'Periodo' if 'Periodo' in df.columns else df.columns[1]
    
    def parse_fecha(p):
        partes = str(p).split('-')
        if len(partes) == 2:
            mes = mes_a_numero_corto(partes[0])
            anio = partes[1]
            if len(anio) == 2:
                anio = "20" + anio
            return f"{anio}-{mes}-01"
        return None
        
    df_adecuado['fecha'] = pd.to_datetime(df_adecuado[col_periodo].apply(parse_fecha))
    
    col_nacional = 'Nacional'
    if col_nacional not in df_adecuado.columns:
        raise ValueError(f"No se encontro columna '{col_nacional}' en el INEC.")
    df_adecuado['empleo_adecuado'] = df_adecuado[col_nacional].astype(str).str.replace(',', '.').astype(float)
    
    return df_adecuado[['fecha', 'empleo_adecuado']]

def procesar_ipc(filepath):
    print("Procesando índice IPC desde tabla pivotada...")
    # Leer saltando los metadatos (primeras 3 lineas)
    # Suponiendo que la linea 3 (indice 2) es "AÑOS,,,,," y los datos empiezan en la linea 4 (indice 3)
    df = pd.read_csv(filepath, skiprows=3, encoding='latin1', header=None)
    
    # Mantener solo las primeras 13 columnas (Año + 12 meses)
    df = df.iloc[:, :13]
    df.columns = ['anio', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    
    # Eliminar filas donde el año sea nulo
    df = df.dropna(subset=['anio'])
    
    # Derretir (melt) la tabla para tener filas por mes
    df_melt = pd.melt(df, id_vars=['anio'], var_name='mes', value_name='ipc')
    
    # Limpiar nulos (meses futuros del ultimo año)
    df_melt = df_melt.dropna(subset=['ipc'])
    
    # Convertir a enteros para armar la fecha
    try:
        df_melt['anio'] = df_melt['anio'].astype(int)
        df_melt['mes'] = df_melt['mes'].astype(int)
    except ValueError:
        # Por si hay texto basura, lo eliminamos
        df_melt['anio'] = pd.to_numeric(df_melt['anio'], errors='coerce')
        df_melt = df_melt.dropna(subset=['anio'])
        df_melt['anio'] = df_melt['anio'].astype(int)
        df_melt['mes'] = df_melt['mes'].astype(int)
    
    # Crear columna fecha
    df_melt['fecha'] = pd.to_datetime(df_melt['anio'].astype(str) + '-' + df_melt['mes'].astype(str).str.zfill(2) + '-01')
    
    # Asegurar que el ipc sea float
    df_melt['ipc'] = df_melt['ipc'].astype(str).str.replace(',', '.').astype(float)
    
    return df_melt[['fecha', 'ipc']].sort_values('fecha').reset_index(drop=True)

def merge_all():
    tasas_path = 'data/raw/tasas_inec.csv'
    ipc_path = 'data/external/ipc.csv'
    wti_path = 'data/external/wti.csv'
    output_path = 'data/raw/dataset.csv'
    
    df_empleo = procesar_tasas_inec(tasas_path)
    df_ipc = procesar_ipc(ipc_path)
    df_wti = pd.read_csv(wti_path, parse_dates=['fecha'])
    
    # Fusionar
    df_final = pd.merge(df_empleo, df_ipc, on='fecha', how='outer')
    df_final = pd.merge(df_final, df_wti, on='fecha', how='outer')
    
    # Filtrar desde 2015 hasta diciembre de 2025 (evitando meses fantasma de 2026 arrastrados por el WTI)
    df_final = df_final[(df_final['fecha'] >= '2015-01-01') & (df_final['fecha'] <= '2025-12-01')]
    
    df_final = df_final.sort_values('fecha').reset_index(drop=True)
    
    # Marcar interpolados del empleo adecuado
    df_final['es_interpolado'] = df_final['empleo_adecuado'].isnull().astype(int)
    
    # Llenar huecos
    df_final['empleo_adecuado'] = df_final['empleo_adecuado'].interpolate(method='linear')
    df_final['ipc'] = df_final['ipc'].interpolate(method='linear')
    df_final['wti'] = df_final['wti'].ffill() 
    
    # Eliminar cualquier nulo final
    df_final = df_final.dropna()
    
    df_final.to_csv(output_path, index=False)
    print(f"\\n¡Dataset final ensamblado con éxito en {output_path}!")
    print(df_final.head())
    print("\\nInformación de huecos interpolados (es_interpolado=1):")
    print(df_final['es_interpolado'].value_counts())

if __name__ == '__main__':
    merge_all()
