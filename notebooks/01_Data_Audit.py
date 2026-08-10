#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:





# 

# In[ ]:





# 

# In[ ]:


print("Valores Faltantes por Columna:")
print(df.isnull().sum())
print("\nFilas Duplicadas:", df.duplicated().sum())
print("\n


# 

# In[ ]:





# 

# In[ ]:





# 

# In[ ]:





# 

# In[ ]:


def test_stationarity(series):
    series = series.dropna()
    print("--- Augmented Dickey-Fuller (ADF) ---")
    adf = adfuller(series)
    print(f"ADF Statistic: {adf[0]:.4f}")
    print(f"p-value: {adf[1]:.4f}")

    print("\n--- KPSS Test ---")
    kpss_stat = kpss(series, regression='c', nlags="auto")
    print(f"KPSS Statistic: {kpss_stat[0]:.4f}")
    print(f"p-value: {kpss_stat[1]:.4f}")

    print("\n


# 

# In[ ]:





# 

# In[ ]:





# 

# In[ ]:





# 
