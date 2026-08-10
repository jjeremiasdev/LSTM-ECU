import yfinance as yf
import pandas as pd
import os

def download_wti(output_path):
    print("Descargando precios históricos del petróleo WTI (Crude Oil Futures)...")
    # Símbolo de WTI Crude Oil en Yahoo Finance
    wti = yf.Ticker("CL=F")
    
    # Descargar datos mensuales desde 2015 hasta 2026
    df = wti.history(start="2015-01-01", end="2026-12-31", interval="1mo")
    
    if df.empty:
        print("Error: No se pudo descargar la data de Yahoo Finance.")
        return
        
    # Limpiar el dataframe
    df = df.reset_index()
    # Yahoo Finance devuelve 'Date' con timezone, lo convertimos a fecha simple (YYYY-MM-01)
    df['fecha'] = df['Date'].dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()
    
    # Nos quedamos con el precio de cierre (Close)
    df_wti = df[['fecha', 'Close']].rename(columns={'Close': 'wti'})
    
    # Redondear a 2 decimales
    df_wti['wti'] = df_wti['wti'].round(2)
    
    # Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_wti.to_csv(output_path, index=False)
    print(f"¡Éxito! Precios WTI guardados en {output_path}")
    print(df_wti.head())

if __name__ == "__main__":
    output = "data/external/wti.csv"
    download_wti(output)
