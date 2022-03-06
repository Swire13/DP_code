# import pandas
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
# from numpy import array
# from keras.models import Sequential
#
# from pandas import read_json
#
# dataset = pandas.read_json('/home/matej/PycharmProjects/DP_code/data/marsWeather_till_17_1_2022.json')
#
# x_train = dataset.values[]
# y_train = dataset.values[]
# x_test = dataset.values[]
# y_test = dataset.values[]
#
# n_features = 1
#
# model = Sequential()
# model.add(Conv1D(filters = 128, kernel_size = 2, activation = 'relu', input_shape = (x_train.shape[1], n_features)))
# shape = (x_train.shape[1], n_features)
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(1))
# model.compile(optimizer = 'adam', loss = 'mse')
#
# history = model.fit(x_train, y_train, validation_split = 0.3, epochs = 50, batch_size = 20, verbose = 2)
#
# y_predict = model.predict(x_test, verbose=0)
#
