import pandas as pd 
from sklearn.linear_model import LinearRegression 
df = pd.read_csv('peso.csv')

X = df['altura']
Y = df['peso']

X_procesada = X.values.reshape(-1,1)
Y_procesada = Y.values.reshape(-1,1)

modelo = LinearRegression()

modelo.fit(X_procesada,Y_procesada)

altura = 1.50

peso = modelo.predict([[altura]])

puntaje = modelo.score(X_procesada, Y_procesada)
# imprmir el dataframe
# print(df)
# puntaje del proceso
print(puntaje)
# modelo prediccion
print(peso)