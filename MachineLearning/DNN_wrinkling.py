


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
import dask.dataframe as dd

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

path = '/media/max/HDD3/DNS_Data/Planar/NX512/UPRIME5/postProcess_DNN'

case = '16'

data_tensor = dd.read_hdf(join(path,'filter_width_16_DNN.hdf'),key='DNS')
#data_grads = pd.read_pickle(join(path,'filter_width_TOPHAT_%s_grad_LES_tensor.pkl' % case))

# col_names = data_tensor.columns
#
# data_tensor_clip = data_tensor.compute().values.reshape(512,512,512,31)
# data_tensor_clip = data_tensor_clip[16:512-16,16:512-16,16:512-16,:]
#
# data_tensor_clip = data_tensor_clip.reshape((512-2*16)**3,31)
# data_tensor =dd.DataFrame(data_tensor_clip,colums=col_names)

#%%
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

data_tensor=data_tensor[data_tensor['omega_model_planar']>1e-1]
data_tensor=data_tensor[data_tensor['c_bar']<0.99]
data_tensor=data_tensor[data_tensor['c_bar']>0.1].sample(frac=0.3).compute()        # sample from the entire data set

#%%
# compute tensors R, S mag(U) etc.

mag_U = np.sqrt(data_tensor['U_bar'].values**2 + data_tensor['V_bar'].values**2 +data_tensor['W_bar'].values**2)
mag_grad_c = np.sqrt(data_tensor['grad_c_x_LES'].values**2 + data_tensor['grad_c_y_LES'].values**2 +data_tensor['grad_c_z_LES'].values**2)

sum_U = data_tensor['U_bar'].values + data_tensor['V_bar']+data_tensor['W_bar'].values
sum_c = abs(data_tensor['grad_c_x_LES'].values) + abs(data_tensor['grad_c_y_LES'].values) +abs(data_tensor['grad_c_z_LES'].values)

grad_U = np.sqrt(data_tensor['grad_U_x_LES'].values**2 + data_tensor['grad_U_y_LES'].values**2 +data_tensor['grad_U_z_LES'].values**2)
grad_V = np.sqrt(data_tensor['grad_V_x_LES'].values**2 + data_tensor['grad_V_y_LES'].values**2 +data_tensor['grad_V_z_LES'].values**2)
grad_W = np.sqrt(data_tensor['grad_W_x_LES'].values**2 + data_tensor['grad_W_y_LES'].values**2 +data_tensor['grad_W_z_LES'].values**2)

mag_grad_U = np.sqrt(grad_U**2 + grad_V**2 +grad_W**2)
sum_grad_U = abs(grad_U) + abs(grad_V) +abs(grad_W)

gradient_tensor = np.array([
                    [data_tensor['grad_U_x_LES'].values,data_tensor['grad_V_x_LES'].values,data_tensor['grad_W_x_LES'].values],
                    [data_tensor['grad_U_y_LES'].values,data_tensor['grad_V_y_LES'].values,data_tensor['grad_W_y_LES'].values],
                    [data_tensor['grad_U_z_LES'].values,data_tensor['grad_V_z_LES'].values,data_tensor['grad_W_z_LES'].values],
                    ])
# symetric strain
Strain = 0.5*(gradient_tensor + np.transpose(gradient_tensor,(1,0,2)))
#anti symetric strain
Anti =  0.5*(gradient_tensor - np.transpose(gradient_tensor,(1,0,2)))

lambda_1 = np.trace(Strain**2)
lambda_2 = np.trace(Anti**2)
lambda_3 = np.trace(Strain**3)
lambda_4 = np.trace(Anti**2 * Strain)
lambda_5 = np.trace(Anti**2 * Strain**2)

data_tensor['mag_grad_c'] = mag_grad_c
data_tensor['mag_U'] = mag_U
data_tensor['sum_c'] = sum_c
data_tensor['sum_U'] = sum_U
data_tensor['sum_grad_U'] = sum_grad_U
data_tensor['mag_grad_U'] = mag_grad_U

data_tensor['lambda_1'] = lambda_1
data_tensor['lambda_2'] = lambda_2
data_tensor['lambda_3'] = lambda_3
data_tensor['lambda_4'] = lambda_4
data_tensor['lambda_5'] = lambda_5
#%%
features = ['c_bar', 'mag_U', 'mag_grad_U','mag_grad_c','sum_c','omega_model_planar',
                      'lambda_1','lambda_2','lambda_3','lambda_3','lambda_4','SGS_flux','lambda_5','UP_delta']

X_data = data_tensor[features] #[['c_bar', 'omega_model_planar', 'U_bar', 'V_bar', 'W_bar', 'U_prime', 'V_prime', 'W_prime']]
y_data = data_tensor['omega_DNS_filtered'] #/ data_tensor['omega_model_planar']

X_scaled = scaler_X.fit_transform(X_data.values)
y_scaled = scaler_y.fit_transform(y_data.values.reshape(-1,1))

print("These are the features to be trained for:\n",features )

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,shuffle=True,test_size=0.1)

X_test,X_val,y_test,y_val = train_test_split(X_test,y_test,shuffle=True,test_size=0.3)



#%%
# set up a simple model

# ANN parameters
dim_input = X_train.shape[1]
dim_label = y_train.shape[1]

DNN = Sequential([
    tf.keras.layers.Dense(64,input_dim=dim_input,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(264, activation='relu'),
    tf.keras.layers.Dense(264, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1,activation='linear')
])

DNN.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# checkpoint (save the best model based validate loss)
filepath = "./save_model/best_DNN.hdf5"

# check if there are weights
if os.path.isdir(filepath):
    DNN.load_weights(filepath)


epochs=100
batch_size=64#1024


epoch_size = X_train.shape[0]
a = 0
base = 2
clc = 2
for i in range(8):
    a += base * clc ** (i)
print(a)
epochs, c_len = a, base
schedule = SGDRScheduler(min_lr=1e-6, max_lr=1e-4,
                         steps_per_epoch=np.ceil(epoch_size / batch_size),
                         cycle_length=c_len, lr_decay=0.6, mult_factor=clc)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<1e-6):
            print("\nReached loss < 1e-6 so cancelling training!")
            self.model.stop_training = True

loss_callback = myCallback()
earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, min_delta=1e-7)

checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             verbose=0,
                             save_best_only=True,
                             mode='min',
                             save_freq=10000)

callbacks = [schedule,loss_callback,earlyStop_callback,checkpoint]
#%%
DNN.summary()

#%%
history = DNN.fit(X_train,y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  #validation_split=0.1,
                  shuffle=True,
                  callbacks=callbacks,
                  validation_data=(X_val,y_val))

#%%

y_pred = DNN.predict(X_test,batch_size=64)

y_pred_rescale = scaler_y.rescale(y_pred)
y_test_rescale = scaler_y.rescale(y_test)

X_test_rescale = scaler_X.rescale(X_test)

#%%
plt.figure()
plt.scatter(y_pred_rescale,y_test_rescale,s=0.2)
plt.show(block=False)

plt.figure()
plt.scatter(X_test_rescale[:,0],y_test_rescale[:],c='b',s=0.2)
plt.scatter(X_test_rescale[:,0],y_pred_rescale[:],c='r',s=0.2)
#plt.scatter(X_test_rescale[:,0],X_test_rescale[:,1],c=  'k',s=0.2)
plt.legend(['Test data','Prediction','Pfitzner model'])
plt.xlabel('c_bar')
plt.ylabel('omega')
plt.show(block=False)

plt.figure()
plt.semilogy(history.history['loss'],scaley='log')
plt.semilogy(history.history['val_loss'],scaley='log')
plt.show(block=False)