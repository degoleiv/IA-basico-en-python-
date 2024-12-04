import pandas as pd 
import seaborn as sb
# crear un modelo de tipo regresion lineal
from sklearn.linear_model import LinearRegression



df = pd.read_csv('celsius.csv')
# Caracteristicas (X) - Etiqueta (Y)
X = df['celsius']
Y = df['fahrenheit']
# CONVERTIR LAS CARACTERISTICAS A UNA LISTA DE LISTAS DE CARACTERISTICAS [[],[],[]]
X_procesada = X.values.reshape(-1,1)
Y_procesada = Y.values.reshape(-1,1)

modelo = LinearRegression()
# entrenar modelo de datos
modelo.fit(X_procesada, Y_procesada)

grafica_datos = sb.scatterplot(x="celsius", y="fahrenheit", data=df, hue="fahrenheit", palette="coolwarm")

# predecir datos 
celsius = 123
prediccion = modelo.predict([[celsius]])

# evaluar el entrenamiento del modelo
puntaje = modelo.score(X_procesada, Y_procesada)

print(df)
print(grafica_datos)
print(prediccion)
print(puntaje)