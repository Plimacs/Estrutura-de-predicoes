import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

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

# Ordenar os conjuntos de dados por data
dolar = dolar.sort_values(by='Data')
bvsp = bvsp.sort_values(by='Data')
gol = gol.sort_values(by='Data')
petroleo = petroleo.sort_values(by='Data')
ouro = ouro.sort_values(by='Data')

#print(dolar[dolar['Data'].isnull()])
#print(bvsp[bvsp['Data'].isnull()])
#print(gol[gol['Data'].isnull()])
#print(petroleo[petroleo['Data'].isnull()])
#print(ouro[ouro['Data'].isnull()])

# Padronizar o formato da data nos conjuntos de dados de Dólar, BVSP e Gol
petroleo['Data'] = petroleo['Data'].apply(lambda x: datetime.strptime(str(x), "%d%m%Y").strftime('%Y-%m-%d') if len(str(x)) == 8 else None)
ouro['Data'] = ouro['Data'].apply(lambda x: datetime.strptime(str(x), "%d%m%Y").strftime('%Y-%m-%d') if len(str(x)) == 8 else None)

#dolar.to_csv('Dados/Dolar_Trado_Limpo.csv', index=False)
#bvsp.to_csv('Dados/BVSP_Trado_Limpo.csv', index=False)
#gol.to_csv('Dados/Gol_Trado_Limpo.csv', index=False)
#ouro.to_csv('Dados/Ouro_Trado_Limpo.csv', index=False)
#petroleo.to_csv('Dados/Petroleo_Trado_Limpo.csv', index=False)

# Mesclar os dados em um único DataFrame usando a coluna 'Data' como chave
merged_data = pd.merge(gol, dolar[['Data', 'Dolar_Close']], on='Data', how='inner')
merged_data = pd.merge(merged_data, petroleo[['Data', 'Ultimo']], on='Data', how='inner')
merged_data = pd.merge(merged_data, ouro[['Data', 'Ultimo']], on='Data', how='inner')
merged_data = pd.merge(merged_data, bvsp[['Data', 'BVSP_Close']], on='Data', how='inner')

# Remover linhas com valores nulos, se houver
merged_data = merged_data.dropna()

# Dividir os dados em características (X) e alvo (y)
X = merged_data[['Dolar_Close', 'Ultimo_x', 'Ultimo_y', 'BVSP_Close']]
y = merged_data['Gol_Close']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer predições no conjunto de teste
predictions = model.predict(X_test)

# Avaliar o desempenho do modelo
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')