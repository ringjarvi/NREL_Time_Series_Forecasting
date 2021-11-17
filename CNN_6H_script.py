import numpy as np
import pandas as pd
import random 
import math
import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Flatten, InputLayer, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import mean_squared_error, mean_absolute_error

import h5py
import os

def build_cnn(input_shape,conv_layers, filter_size, kernel_size, activation, pool_size, layer_sizes, n1, n2, n3, horizon):
    cnn_model=Sequential()
    cnn_model.add(InputLayer(input_shape=input_shape))
    kernel_size=np.int(kernel_size)
    pool_size=np.int(pool_size)    
    for i in range(conv_layers):
        cnn_model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation=activation))
    cnn_model.add(MaxPooling1D(pool_size=pool_size))
    cnn_model.add(Flatten())
    for j, layer_size in enumerate(layer_sizes):
        cnn_model.add(Dense(layer_size, activation=activation))
    cnn_model.add(Dense(horizon))
    cnn_model.compile(optimizer='adam', loss='mae')
    return cnn_model

def create_features(df, features, label=None):
    """
    Creates time series features from datetime index
    label=predicted target
    Can be used for creating features for load
    or weather model construction, by setting kind to 
    load
    """
    rows=df.shape[1]
    df['date'] = df.index
    df['cos(hour)'] = np.cos(2*np.pi*df['date'].dt.hour/24)
    df['sin(hour)'] = np.sin(2*np.pi*df['date'].dt.hour/24)
    
    df['cos(dayofweek)'] = np.cos(2*np.pi*df['date'].dt.dayofweek/7)
    df['sin(dayofweek)'] = np.sin(2*np.pi*df['date'].dt.dayofweek/7)
    
    df['cos(month)'] = np.cos(2*np.pi*df['date'].dt.month/12)
    df['sin(month)'] = np.sin(2*np.pi*df['date'].dt.month/12)
    
    df['cos(dayofyear)'] = np.cos(2*np.pi*df['date'].dt.dayofyear/365.25)
    df['sin(dayofyear)'] = np.sin(2*np.pi*df['date'].dt.dayofyear/365.25)
    
    df['weekday'] = (df['date'].dt.dayofweek // 5 == 1).astype(int)
    
    X = df[[label, 'cos(hour)', 'sin(hour)','cos(dayofweek)',
                'sin(dayofweek)','cos(month)','sin(month)','cos(dayofyear)', 
                'sin(dayofyear)']]
    for f in features:
        X['{}'.format(f)]=df['{}'.format(f)].copy()
    return X 

def load_data(df, seq_len, train_length,horizon):
    X_train = []
    X_train_pred = []
    y_train = []
    for i in range(seq_len, len(df)-horizon+1):
        X_train.append(df.iloc[i-seq_len : i, :].values)
        y_train.append(df.iloc[i:i+horizon, 0].values)
    for i in range(seq_len, len(df)+1):
        X_train_pred.append(df.iloc[i-seq_len : i, :].values)
    
    #1 last 6189 days are going to be used in test
    y_train=pd.DataFrame(list(map(np.ravel, y_train)))
    X_test = X_train[train_length:]             
    y_test = y_train[train_length:]
    
    #2 first 110000 days are going to be used in training
    X_train = X_train[:train_length]           
    y_train = y_train[:train_length]
    
    #3 convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #4 reshape data to input into RNN models
    X_train = np.reshape(X_train, (train_length, seq_len, df.shape[1]))
    
    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, df.shape[1]))
    
    X_test_pred = np.array(X_train_pred[train_length-84:])
    X_test_pred = np.reshape(X_test_pred, (X_test_pred.shape[0], seq_len, df.shape[1]))
    
    return [X_train, y_train, X_test, y_test, X_test_pred]

load_data_path="/lustre/eaglefs/projects/rjsolfor/NYISO/Data/Load_Data/"
ecmwf_data_path="/lustre/eaglefs/projects/rjsolfor/NYISO/Data/ECMWF_Forecasts/"
forecasts_path="/lustre/eaglefs/projects/rjsolfor/NYISO/Load_Forecasts/Deterministic"
metrics_path="/lustre/eaglefs/projects/rjsolfor/NYISO/Load_Forecasts/Deterministic/Metrics/"
model_weights_path='/lustre/eaglefs/projects/rjsolfor/NYISO/Model_Weights/'

h='6H'
model='CNN'
num_forecasts=5 #how many forecasts are generated to be averaged
ecmwf_forecasts_start=25 #max 51
first_time=False

#Zones=['CAPTIL', 'CENTRL', 'DUNWOD', 'GENESE', 'HUD VL', 'LONGIL', 'MHK VL','MILLWD', 'N.Y.C.', 'NORTH', 'WEST']
Zones=[ 'N.Y.C.']
ecmwf_names=['1dayahead_1_air_temp', '1dayahead_1_dewpoint_temp', '1dayahead_2_air_temp', '1dayahead_2_dewpoint_temp', '1dayahead_3_air_temp', '1dayahead_3_dewpoint_temp', '2dayahead_1_air_temp', '2dayahead_1_dewpoint_temp', '2dayahead_2_air_temp', '2dayahead_2_dewpoint_temp', '2dayahead_3_air_temp', '2dayahead_3_dewpoint_temp', 'intraday1_air_temp', 'intraday1_dewpoint_temp', 'intraday2_air_temp', 'intraday2_dewpoint_temp', 'intraday3_air_temp', 'intraday3_dewpoint_temp']

batch_exp=6
conv_layers=2
filter_size=59
kernel_size=4
pool_size=3
dense_layers=2
lower_forecasts=5
upper_forecasts=6
n1=33
n2=49
n3=56
seq_len=45
activation='elu'
layer_sizes=[n1,n2,n3][:dense_layers]

pred_period=8760 #35040
horizon=12

zone_load_data=pd.read_csv(load_data_path + 'NYISO_ALL_ZONES_FILLED_Attenuated.csv',index_col=0)
zone_load_data.index=pd.to_datetime(zone_load_data.index)
zone_load_data15min=zone_load_data.resample('15min').mean()
zone_load_data1hour=zone_load_data.resample('1H').mean()

temp_metrics=[]
for z in Zones:
    #z=Zones[0]
    ecmwf_data=[]
    filename = ecmwf_data_path + "{}_ECMWF_V2.h5".format(z)
    hf1 = h5py.File(filename, 'r')
    hf1.keys()
    for n in ecmwf_names:
        #print('{}'.format(n))
        ecmwf_data.append(pd.DataFrame(hf1.get('{}'.format(n))).values)
    print(z)

    #how many ECMWF forecasts to cycle through, will save a new file load forecast for each forecast
    if(first_time):
        ecmwf_forecasts_start=0
    for k in range(ecmwf_forecasts_start,51):
        #print('k={}'.format(k))
        ecmwf_mem='ECMWF{}'.format(k)
        zone_data=pd.DataFrame(zone_load_data1hour[z][:-1])
        for i in range(len(ecmwf_data)):
            zone_data[ecmwf_names[i]]=ecmwf_data[i][:,k]
        Xl_train= create_features(zone_data,ecmwf_names,label=z)

        mu=Xl_train[0:8760].mean()
        sd=Xl_train[0:8760].std()
        norm_Xl_train=(Xl_train-mu)/sd
        #X_train is formated by [samples, timesteps, features]=[train_length, seq_len, features]
        train_length=Xl_train.shape[0]-pred_period-seq_len-horizon+1


        X_train, y_train, X_test, y_test, X_test_pred = load_data(norm_Xl_train, seq_len,train_length,horizon)

        o_max_rmse=[]
        o_min_rmse=[]
        o_max_mae=[]
        o_min_mae=[]
        o_max_mbe=[]
        o_min_mbe=[]
        o_max_mape=[]
        o_min_mape=[]
        a_max_rmse=[]
        a_min_rmse=[]
        a_max_mae=[]
        a_min_mae=[]
        a_max_mbe=[]
        a_min_mbe=[]
        a_max_mape=[]
        a_min_mape=[]
        avg_forecast_pred=[]
        avg_forecast=[]

        for j in range(num_forecasts):

            random.seed((j+1)*3)


            input_shape=(X_train.shape[1],X_train.shape[2])
            layer_sizes=[n1,n2,n3][:dense_layers]

            cnn_model=build_cnn(input_shape, conv_layers, filter_size, kernel_size, 
                                    activation, pool_size, layer_sizes, 
                                    n1, n2, n3, horizon)

            callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                  ModelCheckpoint(model_weights_path + '{}_{}_{}_Weights.h5'.format(z,h,model), monitor='val_loss', mode='min',save_best_only=True,save_weights_only=True)]

            batch_size=int(math.pow(2,batch_exp))
            # fit model
            cnn_model.fit(X_train, y_train, 
                              epochs=200,
                              batch_size=batch_size,
                              verbose=0,
                              callbacks=callbacks,
                              validation_split=0.2)

            del cnn_model
            test=build_cnn(input_shape,conv_layers, filter_size, kernel_size, 
                               activation, pool_size, layer_sizes, 
                               n1, n2, n3, horizon)
            test.load_weights(model_weights_path + '{}_{}_{}_Weights.h5'.format(z,h,model))
            yhat = test.predict(X_test, verbose=0)
            yhat_pred = test.predict(X_test_pred, verbose=0)

            avg_forecast_pred.append((yhat_pred*sd[0])+mu[0])
            avg_forecast.append((yhat*sd[0])+mu[0])

        all_avg_forecast=np.sum(avg_forecast,axis=0)/num_forecasts 
        all_avg_forecast_pred=np.sum(avg_forecast_pred,axis=0)/num_forecasts
        dti = pd.date_range('2018-12-28 00:00:00', periods=8856, freq='1H')
        cols=[]
        for i in range(7,13):
            cols.append("{}_n_steps_ahead".format(i))

        cnn_6horizon_forecast=pd.DataFrame(all_avg_forecast_pred[:,6:12], columns=cols, index=dti)
        # base dir
        _dir = forecasts_path       

        # create dynamic name, like "D:\Current Download\Attachment82673"
        _dir = os.path.join(_dir, '{}'.format(z))

        # create 'dynamic' dir, if it does not exist
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        cnn_6horizon_forecast.to_csv(forecasts_path + '/{}/{}_{}_{}_Forceast.csv'.format(z,h,model,ecmwf_mem))
        results=pd.DataFrame({'True_Load':zone_data[z].tail(X_test.shape[0]+horizon).values}, index=zone_data[z].tail(X_test.shape[0]+horizon).index)
        rmse=[]
        mae=[]
        mbe=[]
        mape=[]
        tp=yhat.shape[0]
        for i in range(6,horizon):
            rmse.append(np.sqrt(mean_squared_error(results['True_Load'][i+1:tp+i+1].values, all_avg_forecast[:,i])))
            mae.append(mean_absolute_error(results['True_Load'][i+1:tp+i+1].values, all_avg_forecast[:,i]))
            diff=(all_avg_forecast[:,i]-results['True_Load'][i+1:tp+i+1].values)
            mbe.append(diff.mean())
            mape.append(abs((results['True_Load'][i+1:tp+i+1].values- all_avg_forecast[:,i])/results['True_Load'][i+1:tp+i+1].values).mean())
        print("Averaged model forecasts metrics using ensemble weather data {}".format(k))   
        print("RMSE: Max: {}, Min: {}".format(max(rmse),min(rmse)))
        print("MAE: Max: {}, Min: {}".format(max(mae),min(mae)))
        print("MBE: Max: {}, Min: {}".format(max(mbe),min(mbe)))
        print("MAPE: Max: {}, Min: {}".format(max(mape),min(mape)))
        temp_metrics.append([z, h, model, ecmwf_mem, max(rmse), min(rmse), max(mae), min(mae),max(mbe),min(mbe),max(mape),min(mape)])
        metrics=pd.DataFrame(temp_metrics, columns=['Zone','Horizon','Model','ECMWF_Mem','Max RMSE', 'Min RMSE','Max MAE','Min MAE','Max MBE','Min MBE','Max MAPE','Min MAPE' ])
        metrics.to_csv(metrics_path + 'NYISO_{}_{}_Metrics.csv'.format(model,h))
    first_time=True