import numpy as np
import os
import sys
import time
import pandas as pd 
from tqdm._tqdm_notebook import tqdm_notebook
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LocallyConnected1D

import tensorflow as tf

import random

import matplotlib.pyplot as plt
import gc

#fitting parameters

params = {
    "batch_size": 64,
    "epochs": 1,
    "lr": 0.0010000,
    "time_steps": 64,
    "rec_length": 10000
}


TIME_STEPS = params["time_steps"]
BATCH_SIZE = params["batch_size"]
stime = time.time()
steps_per_epoch = 2000


#functions for timing simulation runtime
def time_it(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(t2 - t1)
        return (t2 - t1), res, func.__name__
    return wrapper


def print_time(text, stime):
    seconds = (time.time()-stime)
    print(text, seconds//60,"minutes : ",np.round(seconds%60),"seconds")

#the dataset length needs to be exact multiple of the batch size
def trim_dataset(mat,batch_size):

    no_of_rows_drop = mat.shape[0]%batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat

#this function is responsible for creating the dataset
def build_timeseries(mat, y_col_index):

    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,3))
    print("dim_0",dim_0)
    counter = 0
    for i in range(dim_0):
        #get rid of contaminated results (ie: end of trace - beginning of trace samples)
        if i%10000 < 9900:
            x[counter] = mat[counter:TIME_STEPS+counter]
            y[counter] = mat[TIME_STEPS+counter, y_col_index:3]
            counter += 1
        else:
            pass            
    print("length of time-series i/o",x.shape,y.shape)
    return x[:counter], y[:counter]


#read in the output of the NEURON simulations, drop false columns, and name every column
data = pd.read_csv('final.txt', sep=" ", header=None)
data.drop([5], axis = 1, inplace=True)
data.columns = ["vm", "ik", "ina", "inp0", "inp1"]


#shift every column to positive values
data["vm"] = data["vm"]+150
data["ina"] = data["ina"]*-1


#as sodium and potassium current traces have and extremely non-normal distribution, 
#but the outlier values are the actually relevant ones (ie. AP related currents),
#we truncate them at a reasonable amplitude
for i, item in enumerate(data["ina"]):
    if item > 0.1:
        data["ina"][i] = ((item-0.1)/190)+0.1

for i, item in enumerate(data["ik"]):
    if item > 0.05:
        data["ik"][i] = ((item-0.05)/75)+0.05

print("from this:")
print(data["vm"].min())
print(abs(data["vm"].max()))

print(data["ik"].min())
print(abs(data["ik"].max()))

print(data["ina"].min())
print(abs(data["ina"].max()))

#normalization of every column between 0 and 1. alternatively, they can be normalized by subtracting mean and dividing with SD
data["vm"] = (data["vm"]-data["vm"].min())/abs(data["vm"].max())
print(data["vm"].max())
data["vm"] = data["vm"]/data["vm"].max()
data["ik"] = (data["ik"]-data["ik"].min())/abs(data["ik"].max())
data["ina"] = (data["ina"]-data["ina"].min())/abs(data["ina"].max())



#here, functions for dataset creation are called for fitting and validation

df_train, df_test = train_test_split(data, train_size=0.9, test_size=0.1, shuffle=False)
print(f"Train size is: {len(df_train)}  -- Test size is: {len(df_test)}")

x_train = np.asarray(df_train)
x_test = np.asarray(df_test)





del data

x_t, y_t = build_timeseries(x_train, 0)


del x_train
del df_train

c = list(zip(x_t, y_t))
random.shuffle(c)
x_t, y_t = zip(*c)
x_t = np.asarray(x_t)
y_t = np.asarray(y_t)
#x_t = x_t[::2]
#y_t = y_t[::2]

x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
print("Batch trimmed size",x_t.shape, y_t.shape)

#this is a must have, otherwise the model thinks that the 3 labels are sequential in time
y_t = np.reshape(y_t, [y_t.shape[0], 1, 3])

x_t = np.array(x_t, dtype='float32')
y_t = np.array(y_t, dtype='float32')


x_temp, y_temp = build_timeseries(x_test, 0)

y_temp = np.reshape(y_temp, [y_temp.shape[0], 1, 3])

x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)

print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)

x_val = np.array(x_val, dtype='float32')
y_val = np.array(y_val, dtype='float32')

del y_temp
del x_test
del df_test

#this model creates the ANN wit randomly initialized values

def create_model_CNN_LSTM2():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(128, 1, activation='relu', kernel_initializer="he_uniform", padding="causal", input_shape=(TIME_STEPS, x_t.shape[2])),
        tf.keras.layers.Conv1D(100, 5, activation='relu', kernel_initializer="he_uniform", padding="causal"),
        tf.keras.layers.Conv1D(50, 1, activation='tanh', kernel_initializer='glorot_uniform', padding="causal"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.RepeatVector(x_t.shape[2]),
        #AveragePooling
        tf.keras.layers.LSTM(128, activation='tanh', kernel_initializer='glorot_uniform', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu', kernel_initializer="he_uniform")),
        tf.keras.layers.Dense(100, activation='selu', kernel_initializer="lecun_uniform"),
        tf.keras.layers.Dropout(rate=0.15),
        tf.keras.layers.Dense(100, activation='selu', kernel_initializer="lecun_uniform"),
        tf.keras.layers.Dropout(rate=0.15),
        tf.keras.layers.Dense(100, activation='selu', kernel_initializer="lecun_uniform"),
        tf.keras.layers.Dropout(rate=0.15),
        tf.keras.layers.Dense(100, activation='selu', kernel_initializer="lecun_uniform"),
        tf.keras.layers.Dropout(rate=0.15),
        tf.keras.layers.Dense(3)])
    model.compile(loss=['mse','mse','mse'], loss_weights=[1,0.7,0.7], optimizer=tf.keras.optimizers.Nadam(clipnorm=1.))
    return model


model = None
try:
    model = pickle.load(open("lstm_model", 'rb'))
    print("Loaded saved model...")
except FileNotFoundError:
    print("Model not found")




#%%

#this is the main chunk of the fitting procedure

is_update_model = True
if model is None or is_update_model:
    from keras import backend as K
    print("Building model...")
    #first, we need to define the model structure by calling create_model_CNN_LSTM2()
    model = create_model_CNN_LSTM2()
    #this is an optional code, as the first time the code is run, there are no trained weights to load
    model.load_weights('CNN_LSTM_ball_active1.h5')
    model.summary()
    
    #checkpoint to stop the fitting procedure if the model is not getting better. set the patience value lower for earlier stop
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=100, min_delta=0.0001)
    
    #model checkpoint for saving the model after training steps. If save_weights_only is True, it will save it to .h5, 
    #if it's Flase, it will save the whole model, with optimizer state as well. this is best at the first round of the cell fitting
    mcp = ModelCheckpoint('CNN_LSTM_ball_active2.h5', monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=True, mode='min', period=1)

    # Not used here. But leaving it here as a reminder for future
    r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, 
                                  verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    
    
    history = model.fit(x_t, y_t, epochs=500, verbose=1, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                        trim_dataset(y_val, BATCH_SIZE)), callbacks=[mcp])
    
    print("Done with training!")


model = create_model_CNN_LSTM2()   #change model checkpoint as well!
model.load_weights("Finished_architectures/CNN_LSTM_ball_active2.h5")

#%%
@time_it
def go1():
    for i in range(100):
        model(np.reshape(x_t[0], [1,64,5]), training=False)

@time_it
def go50():
    for i in range(100):
        model(x_t[:50], training=False)

@time_it
def go5000():
    for i in range(100):
        model(x_t[:5000], training=False)

go1()

for k in range(5):
    go1()
    go50()
    go5000()
    
    



