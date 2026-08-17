import os

def download_wti(output_path):
    print("Descargando precios históricos del petróleo WTI (Crude Oil Futures)...")
    try:
        import pandas as pd
        import yfinance as yf
        wti = yf.Ticker("CL=F")
        df = wti.history(start="2015-01-01", end="2025-12-31", interval="1mo")
        if not df.empty:
            df = df.reset_index()
            df['fecha'] = df['Date'].dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()
            df_wti = df[['fecha', 'Close']].rename(columns={'Close': 'wti'})
            df_wti['wti'] = df_wti['wti'].round(2)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_wti.to_csv(output_path, index=False)
            print(f"¡Éxito! Precios WTI guardados en {output_path}")
            print(df_wti.head())
            return
    except Exception as e:
        print(f"Advertencia/Fallback al descargar de Yahoo Finance: {e}")
        
    if os.path.exists(output_path):
        print(f"Utilizando datos locales preexistentes en {output_path}")
    else:
        raise FileNotFoundError(f"No se pudo descargar de Yahoo Finance ni se encontró {output_path}")

if __name__ == "__main__":
    output = "data/external/wti.csv"
    download_wti(output)
