import pandas as pd
from sklearn import linear_model
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
#import numpy as np

#recofida de daros
datos = pd.read_csv("pisosFuenlabrada.csv") 

result =  datos[1:6]
columnas = datos[0:1]
corr_test = pearsonr(datos['precio'], datos['metros'])

print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])

# regesion lineal 1
regresion = linear_model.LinearRegression()

#transformamsos los datos a numpy 2
X = datos['precio'].values.reshape(-1, 1)
Y = datos['metros']
Z = datos['habitaciones']

#modelo 3

modelo = regresion.fit(X, Y)
print("Interseccion (b)",modelo.intercept_)
print("Coeficiente de correlación (m)",modelo.coef_)


#Pruebas con este dinero que puedo comprar
entrada = [[100000],[200000],[300000],[400000]]
salida = modelo.predict(entrada)



#print(datos.describe())
print("Salida: ", salida)

#graficas
#puntos de pruebas
plt.scatter(entrada,salida, color='red')

#linea
plt.plot(entrada,modelo.predict(entrada), color='green')
#elementos
plt.scatter(datos['precio'],datos['metros'], color='blue')

plt.xlabel('precio', fontsize = 20)
plt.ylabel('metros', fontsize = 20)


 
plt.show()


'''
print(f'Lectura de las 5 primeras lineas \n\r {result} ')
print(f'Las columnas son: {columnas} ')
total_cols = len(datos.Name[1])
print("Numero de columnas: "+str(total_cols))

print('Total de filas:',datos.shape[0])
print (datos.dtypes)
print(datos['Age'].mean()) # media
print(datos['Age'].median()) #mediana
print(datos['Customer spendings'].mean())
print(datos['Customer spendings'].median())

#filtro elimina las ventas menos de 10€
#mixto datos = datos.loc[~((datos['Customer spendings'] == 0) | (datos['Age'] == 0))]
datos = datos[datos['Customer spendings'] < 10]
print(datos)

#elimina las filas duplicadas 
datos = datos.drop_duplicates()
print(datos)


#Elimina vacias
datos = datos.dropna()

print(datos)

# graficos
plt.bar(datos['Country'],datos['Customer spendings'])
plt.show()

#deja solo Country, Age, Gender, Customer spendings
##datos.drop(axis=1, columns='Name')
datos =  datos.drop(axis=1, columns=['Name','Phone number','Email','Address','Postal code','Last date of connection','Last time of connection'])
####['Name','Country','Age','Score','Scholarship'])
print(datos)

# Create DataFrame
data = pd.DataFrame(datos)

# Write to CSV file
data.to_csv("Customers.csv")

# Print the output.
print(data)
'''