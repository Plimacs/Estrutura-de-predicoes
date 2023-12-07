import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Carregar os dados
dolar = pd.read_csv('../Dados/BRL=X.csv', parse_dates=['Date'], index_col='Date')['Close']
bvsp = pd.read_csv('../Dados/^BVSP.csv', parse_dates=['Date'], index_col='Date')['Close']
petroleo = pd.read_csv('../Dados/Petroleo_Tratado.csv', parse_dates=['Date'], index_col='Date')['Close']
ouro = pd.read_csv('../Dados/Ouro_Tratado.csv', parse_dates=['Date'], index_col='Date')['Close']
gol = pd.read_csv('../Dados/Goll4SA.csv', parse_dates=['Date'], index_col='Date')['Close']