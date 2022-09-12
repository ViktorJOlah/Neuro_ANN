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
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LocallyConnected1D
from keras import initializers
from keras.layers import Dense, Dropout, Activation, Flatten, Input, TimeDistributed, Reshape, Permute, Flatten, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Nadam

import tensorflow as tf
from collections import deque

import random

import matplotlib.pyplot as plt
import gc
from sys import getsizeof

import keras
keras.backend.set_learning_phase(0)     #tells the system that the model is not training, it's only predicting, it's supposed to speed up the process

import random
import time                                                


#timing decorator for runtime measurement

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result

    return timed


#dictionary for fitting parameters
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

#as the data is presented in batches, the dataset needs to be exact multiple of the batch size

def trim_dataset(mat,batch_size):

    no_of_rows_drop = mat.shape[0]%batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat

#this function builds the final dataset of x (input) and y (value to be predicted) values

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
data = pd.read_csv('create_sample_for_independent_validation/final.txt', sep=" ", header=None)
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

#normalization of every column between 0 and 1. alternatively, they can be normalized by subtracting mean and dividing with SD
data["vm"] = (data["vm"]-73.2615)/190.1118
data["vm"] = data["vm"]/0.6146399118834286
data["ik"] = (data["ik"]+3.15846e-05)/0.0606687066667
data["ina"] = (data["ina"]-6.1946e-06)/0.104076784210526






#here, functions for dataset creation are called for fitting and validation

df_train, df_test = train_test_split(data, train_size=0.9, test_size=0.1, shuffle=False)
print(f"Train size is: {len(df_train)}  -- Test size is: {len(df_test)}")

x_train = np.asarray(df_train)
x_test = np.asarray(df_test)

del data

x_t, y_t = build_timeseries(x_train, 0)


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




####################################################################################################### ARCHITECTURES ##########################################################################

#################################### LINEAR ####################################

def create_model_linear():
    model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(TIME_STEPS, x_t.shape[2])),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(3)
            ])
    model.compile(loss=['mse'], loss_weights=[1], optimizer=tf.keras.optimizers.Nadam(clipnorm=1.))
    return model


#################################### NONLINEAR #################################### 

def create_model_nonlinear():
    model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(TIME_STEPS, x_t.shape[2])),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='sigmoid'),
            tf.keras.layers.Dense(3)
            ])
    model.compile(loss=['mse'], loss_weights=[1], optimizer=tf.keras.optimizers.Nadam(clipnorm=1.))
    return model

#################################### DNN ####################################

def create_model_DNN():
    model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(TIME_STEPS, x_t.shape[2])),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(128, activation='tanh'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(256, activation='tanh'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(512, activation='tanh'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(1024, activation='tanh'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(1024, activation='tanh'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(512, activation='tanh'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(256, activation='tanh'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(128, activation='tanh'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(3)
            ])
    model.compile(loss=['mse'], optimizer=tf.keras.optimizers.Nadam(clipnorm=1.))
    return model

#################################### WAVENET ####################################

num_segments=1
num_syn_types=2
filter_sizes_per_layer=[5] * 3
num_filters_per_layer=[64] * 3
activation_function_per_layer=['relu'] * 3
l2_regularization_per_layer=[1e-8] * 3
strides_per_layer=[1] * 3
dilation_rates_per_layer=[1] * 3
initializer_per_layer=[0.002] * 3
from keras.regularizers import l1,l2,l1_l2

def create_model_wavenet(TIME_STEPS, num_segments, num_syn_types, filter_sizes_per_layer,
                                                                                            num_filters_per_layer,
                                                                                            activation_function_per_layer,
                                                                                            l2_regularization_per_layer,
                                                                                            strides_per_layer,
                                                                                            dilation_rates_per_layer,
                                                                                            initializer_per_layer):
    
    # define input and flatten it
    binary_input_mat = Input(shape=(TIME_STEPS, x_t.shape[2]), name='input_layer')

    for k in range(len(filter_sizes_per_layer)):
        num_filters   = num_filters_per_layer[k]
        filter_size   = filter_sizes_per_layer[k]
        activation    = activation_function_per_layer[k]
        l2_reg        = l2_regularization_per_layer[k]
        stride        = strides_per_layer[k]
        dilation_rate = dilation_rates_per_layer[k]
        initializer   = initializer_per_layer[k]
        
            
        if not isinstance(initializer, str):
            initializer = initializers.TruncatedNormal(stddev=initializer)
        
        if k == 0:
            x = Conv1D(num_filters, filter_size, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(l2_reg),
                       strides=stride, dilation_rate=dilation_rate, padding='causal', name='layer_%d' %(k + 1))(binary_input_mat)
        else:
            x = Conv1D(num_filters, filter_size, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(l2_reg),
                       strides=stride, dilation_rate=dilation_rate, padding='causal', name='layer_%d' %(k + 1))(x)
        x = BatchNormalization(name='layer_%d_BN' %(k + 1))(x)
        

    output_soma_init  = initializers.TruncatedNormal(stddev=0.03)
    intermediate = Conv1D(1, 1, activation='linear' , kernel_initializer=output_soma_init, kernel_regularizer=l2(1e-8), padding='causal', name='somatic')(x)
    flat = Flatten()(intermediate)
    output_soma_voltage_pred = Dense(3)(flat)

    temporaly_convolutional_network_model = Model(inputs=binary_input_mat, outputs=output_soma_voltage_pred)

    optimizer_to_use = Nadam(lr=0.0001)
    temporaly_convolutional_network_model.compile(optimizer=optimizer_to_use,
                                                  loss='mse')
    
    return temporaly_convolutional_network_model



#################################### CNN_LSTM ####################################

def create_model_CNN_LSTM():
    cnn_lstm = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(128, 1, activation='relu', kernel_initializer="he_uniform", padding="causal", input_shape=(TIME_STEPS, x_t.shape[2])),
            tf.keras.layers.Conv1D(100, 5, activation='relu', kernel_initializer="he_uniform", padding="causal"),
            tf.keras.layers.Conv1D(50, 1, activation='tanh', kernel_initializer='glorot_uniform', padding="causal"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.RepeatVector(x_t.shape[2]),
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
    cnn_lstm.compile(loss=['mse'], optimizer=tf.keras.optimizers.Nadam(clipnorm=1.))
    return cnn_lstm



##############################################################################  create models ###############################################################################


linear = create_model_linear()
linear.load_weights('Finished_architectures/ball_act_linear.h5')
print("linear OK")

nonlinear = create_model_nonlinear()
nonlinear.load_weights('Finished_architectures/ball_act_nonlinear.h5')
print("nonlinear OK")
DNN = create_model_DNN()
DNN.load_weights('Finished_architectures/ball_act_DNN.h5')
print("DNN OK")

wavenet = create_model_wavenet(TIME_STEPS, 1, num_syn_types,
                                    filter_sizes_per_layer, num_filters_per_layer,
                                    activation_function_per_layer, l2_regularization_per_layer,
                                    strides_per_layer, dilation_rates_per_layer, initializer_per_layer)
wavenet.load_weights('Finished_architectures/ball_act_wavenet.h5')
print("wavenet OK")
CNN_LSTM = create_model_CNN_LSTM()
CNN_LSTM.load_weights('Finished_architectures/CNN_LSTM_ball_active2.h5')
print("CNN_LSTM OK")

@tf.function
def serve_CNN_LSTM(x):
    return CNN_LSTM(x, training=False)

@tf.function
def serve_wavenet(x):
    return wavenet(x, training=False)

@tf.function
def serve_DNN(x):
    return DNN(x, training=False)

@tf.function
def serve_nonlinear(x):
    return nonlinear(x, training=False)

@tf.function
def serve_linear(x):
    return linear(x, training=False)

x_t = np.array(x_t, dtype='float32')

#this needs to be done by hand as it is too much for memory
"""
linear_fit = serve_linear(x_t[:50000])
nonlinear_fit = serve_nonlinear(x_t[:50000])
DNN_fit = serve_DNN(x_t[:50000])
wavenet_fit = serve_wavenet(x_t[:50000])
CNN_LSTM_fit = serve_CNN_LSTM(x_t[:50000])
"""

#test continous prediction



#%%
scale1 = 0.96938
scale2 = 0.00241
scale3 = 1.10493
scale4 = 0.00179
scale5 = 0.97887
scale6 = 0.00893

#%%

from sklearn.linear_model import LinearRegression


#this function runs continous prediction for 500 ms

def go():
    #to get an average of the fit results, we simulate 50 models at the same time
    for i in range(50):
        out = []
        #every model receives initialization values from the loaded simulation, 2 seconds apart
        starting_point = i*2000
        updated = deque(x_t[starting_point], maxlen=64)
        counter = starting_point
        for _ in range(500):
            #calling the ANN and reshaping the output for further processing
            new_mp = (np.mean(np.asarray(serve_CNN_LSTM(np.reshape(np.asarray(updated), [1,64,5]).astype('float32')))[0], axis=0)/0.91)-0.008
            #saving the output of the model
            out.append(new_mp)
            counter += 1
            #creating new input by updating the previous input with the predicted values and the synapse timings from the NEURON simulation
            new_input = np.asarray([new_mp[0]/scale1-scale2, new_mp[1]/scale3-scale4, new_mp[2]/scale5-scale6, x_t[counter][-1][3], x_t[counter][-1][4]])
            updated.append(new_input)
        
        #the two final vectors are the prediction (vm_out) and the corresponding ground truth values (vm_GT)
        vm_out = np.transpose(out)[0]
        vm_GT = np.transpose(np.asarray(y_t[starting_point:counter]))[0][0]
      
        mse = np.mean((vm_out-vm_GT)**2)
        var = np.var(vm_GT)
        var_explained = 1-(mse/var)
        reg = LinearRegression().fit(np.reshape(vm_GT, [-1, 1]), np.reshape(vm_out, [-1, 1]))
        r_square = reg.score(np.reshape(vm_GT, [-1, 1]), np.reshape(vm_out, [-1, 1]))
        r_value = np.sqrt(r_square)
        #print(f"variance explained is: {var_explained}, regression coefficient is {r_value}")
        print(f"{var_explained}  {r_value}")



go()
