#%%
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

from keras.models import Sequential
from collections import deque




import keras

import tensorflow as tf

import random

import matplotlib.pyplot as plt
import gc




#%%
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




def print_time(text, stime):
    seconds = (time.time()-stime)
    print(text, seconds//60,"minutes : ",np.round(seconds%60),"seconds")


def trim_dataset(mat,batch_size):

    no_of_rows_drop = mat.shape[0]%batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat

def build_timeseries(mat, y_col_index):

    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,8))
    print("dim_0",dim_0)
    counter = 0
    for i in range(dim_0):
        #get rid of contaminated results (ie: end of trace - beginning of trace samples)
        if i%10001 < 9900:
            x[counter] = mat[counter:TIME_STEPS+counter]
            y[counter] = mat[TIME_STEPS+counter, y_col_index:8]
            counter += 1
        else:
            pass            
    print("length of time-series i/o",x.shape,y.shape)
    return x[:counter], y[:counter]



#%%

data = pd.read_csv('results/vmi.txt', sep=" ", header=None, dtype='float32')
data.drop([208], axis = 1, inplace=True)
data.columns = [str(data.columns[i]) for i in range(208)]
data["0"] = data["0"]+150   #somatic Vm
data["4"] = data["4"]+150   #tuft Vm
data["6"] = data["6"]+150   #axonal Vm
data["2"] = data["2"]*-1    #somatic sodium
data["3"] = data["3"]*-1    #somatic calcium
data["5"] = data["5"]*-1    #tuft calcium
data["7"] = data["7"]*-1    #axonal calcium


#%%
for i, item in enumerate(data["2"]):        #somatic sodium scaling    
    if item > 0.025:
        data["2"][i] = ((item-0.025)/15)+0.025

for i, item in enumerate(data["3"]):        #somatic calcium scaling
    if item > 0.0002:
        data["3"][i] = ((item-0.0002)/190)+0.0002

#tuft calcium doesn't need scaling

for i, item in enumerate(data["7"]):        #axonal calcium scaling
    if item > 0.000005:
        data["7"][i] = ((item-0.000005)/190)+0.000005

for i, item in enumerate(data["1"]):        #somatic potassium scaling
    if item > 0.05:
        data["1"][i] = ((item-0.05)/75)+0.05



#first normalize data["vm"]
data["0"] = (data["0"]-72.56649780273438)/170.4821014404297
data["0"] = data["0"]/0.5743453583126393

data["4"] = (data["4"]-78.381103515625)/112.7384033203125
data["4"] = data["4"]/0.30475240727927905

data["6"] = (data["6"]-56.268898010253906)/175.835205078125
data["6"] = data["6"]/0.6799907163912188


data["1"] = (data["1"]-0.0002694719878491014)/0.055874425917863846
data["2"] = (data["2"]-0.0001439540064893663)/0.1006593406200409
data["3"] = (data["3"]-4.793540028913412e-06)/0.00021110274246893823
data["5"] = (data["5"]-4.2306099203415215e-05)/0.0016568000428378582
data["7"] = (data["7"]-1.23903996465018e-14)/1.5140210052777547e-05

split_size = 0.9
batch_per_epoch = data.shape[0]*split_size*0.9

df_train, df_test = train_test_split(data, train_size=split_size, test_size=1-split_size, shuffle=False)
print(f"Train size is: {len(df_train)}  -- Test size is: {len(df_test)}")

x_train = np.asarray(df_train, dtype="float32")
x_test = np.asarray(df_test, dtype="float32")




#%%

def create_model_CNN_LSTM2():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(2048, 1, activation='relu', kernel_initializer="he_uniform", padding="causal", input_shape=(TIME_STEPS, 208)),
        tf.keras.layers.Conv1D(128, 5, activation='relu', kernel_initializer="he_uniform", padding="causal"),
        tf.keras.layers.Conv1D(64, 1, activation='tanh', kernel_initializer='glorot_uniform', padding="causal"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.RepeatVector(208),
        tf.keras.layers.LSTM(128, activation='tanh', kernel_initializer='glorot_uniform', return_sequences=True),
        tf.keras.layers.LSTM(128, activation='tanh', kernel_initializer='glorot_uniform', return_sequences=True),
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
        tf.keras.layers.Dense(8)])
    model.compile(loss=['mse','mse','mse','mse','mse','mse','mse','mse'], loss_weights=[1,0.7,0.7,0.7,1,0.7,0.7,0.7], optimizer=tf.keras.optimizers.Nadam())
    return model


model = None
try:
    model = pickle.load(open("lstm_model", 'rb'))
    print("Loaded saved model...")
except FileNotFoundError:
    print("Model not found")
#%%


model = create_model_CNN_LSTM2()
model.load_weights("CNN_LSTM_L5_active.h5")
# %%

@tf.function
def serve_CNN_LSTM(x):
    return model(x, training=False)


def print_time(text, stime):
    seconds = (time.time()-stime)
    print(text, seconds//60,"minutes : ",np.round(seconds%60),"seconds")


def trim_dataset(mat,batch_size):

    no_of_rows_drop = mat.shape[0]%batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat

def build_timeseries(mat, y_col_index):

    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,8))
    print("dim_0",dim_0)
    counter = 0
    for i in range(dim_0):
        #get rid of contaminated results (ie: end of trace - beginning of trace samples)
        if i%10000 < 9900:
            x[counter] = mat[counter:TIME_STEPS+counter]
            y[counter] = mat[TIME_STEPS+counter, y_col_index:8]
            counter += 1
        else:
            pass            
    print("length of time-series i/o",x.shape,y.shape)
    return x[:counter], y[:counter]


#%%

x_train = np.asarray(data)
del data
x_t, y_t = build_timeseries(x_train, 0)


#%%
from sklearn.linear_model import LinearRegression
def go():
    for i in range(1):
        out = []
        #starting_point = i*2000
        starting_point = i*2000
        updated = deque(x_t[starting_point], maxlen=64)
        counter = starting_point
        for i101 in range(500):
            print(i101)
            new_mp1 = (np.mean(np.asarray(serve_CNN_LSTM(np.reshape(np.asarray(updated), [1,64,208]).astype('float32')))[0], axis=0))
            new_mp = [element*9 for element in new_mp1]
            out.append(new_mp)
            counter += 1
            new_input = x_t[counter][-1]
            for k_place, k_val in enumerate(new_mp):
                new_input[k_place] = k_val
            #new_input = np.asarray([new_mp[0], new_mp[1], new_mp[2], x_t[counter][-1][3], x_t[counter][-1][4]])
            updated.append(new_input)
        
        vm_out = np.transpose(out)[0]
        vm_GT = np.transpose(np.asarray(y_t[starting_point:counter]))[0][0]
        plt.plot(vm_out)
        plt.show()
        """
        mse = np.mean((vm_out-vm_GT)**2)
        var = np.var(vm_GT)
        var_explained = 1-(mse/var)
        reg = LinearRegression().fit(np.reshape(vm_GT, [-1, 1]), np.reshape(vm_out, [-1, 1]))
        r_square = reg.score(np.reshape(vm_GT, [-1, 1]), np.reshape(vm_out, [-1, 1]))
        r_value = np.sqrt(r_square)
        #print(f"variance explained is: {var_explained}, regression coefficient is {r_value}")
        print(f"{var_explained}  {r_value}")
        """


#%%
go()



# %%
CNN_LSTM_fit = serve_CNN_LSTM(x_t[:1500])
# %%
