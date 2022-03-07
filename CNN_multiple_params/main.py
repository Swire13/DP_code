#source: https://medium.com/analytics-vidhya/weather-forecasting-with-recurrent-neural-networks-1eaa057d70c3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset from data.csv file
dataset = pd.read_json('/home/matej/PycharmProjects/DP_code/data/marsWeather_till_17_1_2022.json')

dataset = dataset.dropna(subset=["max_temp"])
dataset=dataset.reset_index(drop=True)

training_set = dataset.iloc[:,5:8].values # all rows, 6th column (max_temp column)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []
n_future = 4 # next 4 days temperature forecast
n_past = 30 # Past 30 days - it has to be divided by mars month table (months are complicated in mars...)
for i in range(0,len(training_set_scaled)-n_past-n_future+1):
    x_train.append(training_set_scaled[i : i + n_past , 0])
    y_train.append(training_set_scaled[i + n_past : i + n_past + n_future , 0 ])
x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1) )

import tensorflow
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional

# Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.

regressor = Sequential()
regressor.add(Bidirectional(LSTM(units=30, return_sequences=True, input_shape = (x_train.shape[1],1) ) ))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30 , return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30 , return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 30))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = n_future,activation='linear'))
regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
history = regressor.fit(x_train, y_train, epochs=500,batch_size=32 )

# # read test dataset
# testdataset = pd.read_csv('data (12).csv')
# #get only the temperature column
# testdataset = testdataset.iloc[:30,3:4].values
# real_temperature = pd.read_csv('data (12).csv')
# real_temperature = real_temperature.iloc[30:,3:4].values
# testing = sc.transform(testdataset)
# testing = np.array(testing)
# testing = np.reshape(testing,(testing.shape[1],testing.shape[0],1))

# predicted_temperature = regressor.predict(testing)
# predicted_temperature = sc.inverse_transform(predicted_temperature)
# predicted_temperature = np.reshape(predicted_temperature,(predicted_temperature.shape[1],predicted_temperature.shape[0]))

