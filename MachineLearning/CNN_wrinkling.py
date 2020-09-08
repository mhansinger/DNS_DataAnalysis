

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
import sklearn as sk
import scipy as sc
import os
from os.path import join
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.customObjects import coeff_r2, SGDRScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

#%%

'''
Load the data
'''

path = '/media/max/HDD3/DNS_Data/Planar/NX512/UPRIME5/postProcess_UVW'

case = '16'

data_tensor = pd.read_pickle(join(path,'filter_width_TOPHAT_16_UWV_Junsu.pkl'))
#data_grads = pd.read_pickle(join(path,'filter_width_TOPHAT_%s_grad_LES_tensor.pkl' % case))

# #%%
#
# print(data_tensor.columns)
# print(data_grads.columns)
#
# #%%
#
# data_tensor_dd = dd.from_pandas(data_tensor,npartitions=1)
# data_grads_dd = dd.from_pandas(data_grads,npartitions=1)
#
# data_all = dd.concat([data_grads_dd,data_tensor_dd],axis=1)

class myStandardScaler():
    def __init__(self):
        self.mean=None
        self.var = None

    def fit_transform(self,data,label=True):
        try:
            assert type(data) is np.ndarray
        except AssertionError:
            print('Only numpy arrays!')

        if label is True:
            self.mean = data.mean()
            self.std = data.std()
        else:
            self.mean = data.mean(axis=1).reshape(-1, 1)
            self.std = data.std(axis=1).reshape(-1, 1)

        transformed = (data - self.mean)/self.std

        return transformed

    def rescale(self,data):
        try:
            assert type(data) is np.ndarray
        except AssertionError:
            print('Only numpy arrays!')

        rescaled = data * self.std + self.mean

        return rescaled


label_name = 'omega_DNS_filtered'

col_names = data_tensor.columns

#%%
scaler_X = myStandardScaler()
scaler_y = myStandardScaler()

# data_tensor=data_tensor[data_tensor['c_bar']<0.95]
# data_tensor=data_tensor[data_tensor['c_bar']>0.05]

X_data = data_tensor[['c_bar', 'omega_model_planar', 'U_bar', 'V_bar', 'W_bar', 'U_prime', 'V_prime', 'W_prime']]
y_data = data_tensor['omega_DNS_filtered']

X_scaled = scaler_X.fit_transform(X_data.values)
y_scaled = scaler_y.fit_transform(y_data.values.reshape(-1,1))

X_3D = X_scaled.reshape(100,100,100,8)
y_3D = y_scaled.reshape(100,100,100,1)

# pos = 50
#
# X_train = X_3D[:,:,:pos,:].reshape(pos*100*100,8)
# X_test = X_3D[:,:,pos:,:].reshape(100-pos*100*100,8)
#
# y_train = y_3D[:,:,:pos,:].reshape(pos*100*100,1)
# y_test = y_3D[:,:,pos:,:].reshape(100-pos*100*100,1)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,shuffle=True,test_size=0.5)




#%%
# set up a simple model

# ANN parameters
dim_input = X_train.shape[1]
dim_label = y_train.shape[1]

DNN = Sequential([
    tf.keras.layers.Dense(64,input_dim=dim_input,activation='relu'),
    tf.keras.layers.Dense(128,input_dim=dim_input,activation='relu'),
    tf.keras.layers.Dense(64,input_dim=dim_input,activation='relu'),
    tf.keras.layers.Dense(1,activation='linear')
])

DNN.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# checkpoint (save the best model based validate loss)
filepath = "./save_model/best_DNN.hdf5"

# check if there are weights
if os.path.isdir(filepath):
    DNN.load_weights(filepath)


epochs=100
batch_size=1024


epoch_size = X_train.shape[0]
a = 0
base = 2
clc = 2
for i in range(9):
    a += base * clc ** (i)
print(a)
epochs, c_len = a, base
schedule = SGDRScheduler(min_lr=1e-6, max_lr=1e-4,
                         steps_per_epoch=np.ceil(epoch_size / batch_size),
                         cycle_length=c_len, lr_decay=0.6, mult_factor=clc)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<1e-5):
            print("\nReached loss < 1e-5 so cancelling training!")
            self.model.stop_training = True

loss_callback = myCallback()

callbacks = [schedule,loss_callback]
#%%
DNN.summary()

#%%
history = DNN.fit(X_train,y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.1,
                  shuffle=True,
                  callbacks=callbacks)

#%%

y_pred = DNN.predict(X_test)

y_pred_rescale = scaler_y.rescale(y_pred)
y_test_rescale = scaler_y.rescale(y_test)

X_test_rescale = scaler_X.rescale(X_test)

#%%
plt.scatter(y_pred_rescale,y_test_rescale,s=0.2)
plt.show()


plt.scatter(X_test_rescale[:,0],y_test_rescale[:],c='b',s=0.2)
plt.scatter(X_test_rescale[:,0],y_pred_rescale[:],c='r',s=0.2)
plt.scatter(X_test_rescale[:,0],X_test_rescale[:,1],c='k',s=0.2)
plt.show()


plt.semilogy(history.history['loss'],scaley='log')
plt.semilogy(history.history['val_loss'],scaley='log')
plt.show()