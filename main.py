# Importando as bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt

# Constantes para definir o intervalo de datas desejado (filtro de intervalo de datas)
#data_inicio = '2023-01-01'
#data_fim = '2023-10-30'

# Carregar os dados
dolar = pd.read_csv('Dados/BRL=X.csv')
bvsp = pd.read_csv('Dados/^BVSP.csv')
petroleo = pd.read_csv('Dados/Petroleo_Tratado.csv')
ouro = pd.read_csv('Dados/Ouro_Tratado.csv')
gol = pd.read_csv('Dados/GOLL4.SA.csv')

# Padronizar os nomes das colunas
dolar = dolar.rename(columns={'Date': 'Data', 'Close': 'Dolar_Close'})
bvsp = bvsp.rename(columns={'Date': 'Data', 'Close': 'BVSP_Close'})
gol = gol.rename(columns={'Date': 'Data', 'Close': 'Gol_Close'})
petroleo = petroleo.rename(columns={'Ultimo': 'Petroleo_Close'})
ouro = ouro.rename(columns={'Ultimo': 'Ouro_Close'})

# Ordenar os conjuntos de dados por data
dolar = dolar.sort_values(by='Data')
bvsp = bvsp.sort_values(by='Data')
gol = gol.sort_values(by='Data')
petroleo = petroleo.sort_values(by='Data')
ouro = ouro.sort_values(by='Data')

# Padronizar o formato da data nos conjuntos de dados de Dólar, BVSP e Gol
petroleo['Data'] = petroleo['Data'].apply(lambda x: datetime.strptime(str(x), "%d%m%Y").strftime('%Y-%m-%d') if len(str(x)) == 8 else None)
ouro['Data'] = ouro['Data'].apply(lambda x: datetime.strptime(str(x), "%d%m%Y").strftime('%Y-%m-%d') if len(str(x)) == 8 else None)

# Filtrar os dados no intervalo de datas desejado (filtro de intervalo de datas)
#dolar = dolar[(dolar['Data'] >= data_inicio) & (dolar['Data'] <= data_fim)]
#bvsp = bvsp[(bvsp['Data'] >= data_inicio) & (bvsp['Data'] <= data_fim)]
#gol = gol[(gol['Data'] >= data_inicio) & (gol['Data'] <= data_fim)]
#petroleo = petroleo[(petroleo['Data'] >= data_inicio) & (petroleo['Data'] <= data_fim)]
#ouro = ouro[(ouro['Data'] >= data_inicio) & (ouro['Data'] <= data_fim)]

# Mesclar os dados em um único DataFrame usando a coluna 'Data' como chave
merged_data = pd.merge(gol, dolar[['Data', 'Dolar_Close']], on='Data', how='inner')
merged_data = pd.merge(merged_data, petroleo[['Data', 'Petroleo_Close']], on='Data', how='inner')
merged_data = pd.merge(merged_data, ouro[['Data', 'Ouro_Close']], on='Data', how='inner')
merged_data = pd.merge(merged_data, bvsp[['Data', 'BVSP_Close']], on='Data', how='inner')

# Remover linhas com valores nulos
merged_data = merged_data.dropna()

# Dividir os dados em características (X) e alvo (y)
X = merged_data[['Dolar_Close', 'Petroleo_Close', 'Ouro_Close', 'BVSP_Close']]
y = merged_data['Gol_Close']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer predições no conjunto de teste
predict = model.predict(X_test)

# Avaliar o desempenho do modelo
print(f'Mean Absolute Error: {mean_absolute_error(y_test, predict)}')
print(f'Root Mean Squared Error: {mean_squared_error(y_test, predict, squared=False)}')

# Criar o modelo RandomForestRegressor
#random_forest_model = RandomForestRegressor(random_state=42)

# Treinar o modelo RandomForestRegressor
#random_forest_model.fit(X_train, y_train)

# Fazer predições no conjunto de teste
#forest_predict = random_forest_model.predict(X_test)

# Modelo RandomForestRegressor
#print("Random Forest:")
#print(f'Mean Absolute Error: {mean_absolute_error(y_test, forest_predict)}')
#print(f'Root Mean Squared Error: {mean_squared_error(y_test, forest_predict, squared=False)}')

# Plotando gráfico para dados de treinamento
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, model.predict(X_train), color='blue')
plt.title('Previsões vs. Valores Reais (Treinamento)')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')

# Plotando gráfico para dados de teste
plt.subplot(1, 2, 2)
plt.scatter(y_test, predict, color='red')
plt.title('Previsões vs. Valores Reais (Teste)')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')

# Ajusta e exibe o gráfico
plt.tight_layout()
plt.show()