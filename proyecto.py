import numpy as np
np.random.seed(4)
import pandas as pd

# Lectura de los datos
#dataset = pd.read_csv('Apple.csv', index_col='Date', parse_dates=['Date'])
#dataset = pd.read_csv('Amazon.csv', index_col='Date', parse_dates=['Date'])
dataset = pd.read_csv('Facebook.csv', index_col='Date', parse_dates=['Date'])
dataset.head()

# Sets de entrenamiento y validación 
# La LSTM se entrenará con datos de 15/10/2020 hacia atrás. La validación se hará con datos de 16/10/2020 hasta 11/12/2020
# En ambos casos sólo se usará el valor más alto de la acción para cada día
set_entrenamiento = dataset[:'2020'].iloc[:,1:2]
set_validacion = dataset['2020':].iloc[:,1:2]

# Normalización del set de entrenamiento
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 5
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshape X_train para que se ajuste al modelo en Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Red LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM
dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 50
modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse')
modelo.fit(X_train,Y_train,epochs=20,batch_size=32)

# Validación (predicción del valor de las acciones)
x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)

# Graficar resultados
import matplotlib.pyplot as plt

def graficar_predicciones(real, prediccion):
    plt.plot(real[0:len(prediccion)],color='red', label='Valor real de la accion')
    plt.plot(prediccion, color='blue', label='Prediccion de la accion')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la accion')
    plt.legend()
    plt.show()
     
graficar_predicciones(set_validacion.values,prediccion)
