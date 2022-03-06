# import inline as inline
# import matplotlib
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import os
# import datetime as dt
# import matplotlib.pyplot as plt
#
# df=pd.read_json("/content/drive/MyDrive/fei/DP/marsWeather_till_17_1_2022.json")
#
# df['terrestrial_date']=pd.to_datetime(df['terrestrial_date'])
# df.terrestrial_date.head()
#
# %matplotlib inline
# plt.figure(figsize=(20,10))
# df.iloc[:,6].value_counts(normalize=True,sort=True).plot(kind='bar')
# plt.xlabel('Max Temperature in degree Celsuis')
# plt.ylabel('% percentage')
# plt.show()
#
# df.iloc[:,6].isna().sum()
#
# df['month']=pd.to_datetime(df.terrestrial_date).dt.to_period('M')
# df.month.value_counts()
#
# aa=df.iloc[:,6].name
# ac=df.groupby(by='month')[aa].mean()
# ab=df['max_temp'].isna()
# df.loc[ab,['max_temp']]=df.loc[ab,'month'].apply(lambda x:ac[x])
# df['max_temp'].isna().sum()
#
# new_df=pd.DataFrame(list(df['max_temp']),index=df.terrestrial_date,columns=['Maxtemperature'])
#
# new_df=new_df.resample('D').mean()
#
# month_df=new_df.resample('M').mean()
#
# year_df=new_df.resample('Y').mean()
#
# %matplotlib inline
# plt.figure(figsize=(20,10))
# plt.plot(new_df)
# plt.show()
#
# from keras.layers import Bidirectional,LSTM,Dense,Flatten,Conv1D,MaxPooling1D,Dropout,RepeatVector
# from keras.models import Sequential
# from keras.callbacks import EarlyStopping,ReduceLROnPlateau
#
# early_stop=EarlyStopping(monitor='loss',patience=5)
#
# n_timesteps=30
# n_features=1
# model=Sequential([LSTM(256, activation='relu', input_shape=(n_timesteps, n_features)),
#                   Dense(128, activation='relu'),
#                   Dense(1)])
# model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
#
# new_df1=pd.DataFrame(list(df['max_temp']), index=df['terrestrial_date'], columns=['temp'])
#
# new_df1=new_df1.resample('D').mean()
# new_df1.temp.isna().sum()
#
# new_df1.fillna(df['max_temp'].mean(),inplace=True)
# new_df1.temp.isna().sum()
#
# from sklearn.preprocessing import MinMaxScaler
#
# scaler=MinMaxScaler(feature_range=(-1,1))
#
# scaled_data=scaler.fit_transform(new_df1)
# scaled_data[:5]
#
# steps=30
# inp1=[]
# out1=[]
#
# for i in range(len(scaled_data)-steps):
#     inp1.append(scaled_data[i:i+steps])
#     out1.append(scaled_data[i+steps])
#
# inp1=np.asanyarray(inp1)
# out1=np.asanyarray(out1)
# x_train1=inp1[:1500,:,:]
# x_test1=inp1[1500:,:,:]
# y_train1=out1[:1500]
# y_test1=out1[1500:]
#
# history = model.fit(x_train1,y_train1,epochs=20)
#
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.show()
#
# predicted=model.predict(x_test1)
# predicted1=scaler.inverse_transform(predicted)
# y_test2=scaler.inverse_transform(y_test1)
#
# plt.figure(figsize=(20,5))
# plt.plot(predicted1,'r',label='predicted')
# plt.plot(y_test2,'g',label='actual')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(20,5))
# plt.plot(y_test2,'g',label='actual')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(20,5))
# plt.plot(predicted1,'r',label='predicted')
# plt.legend()
# plt.show()