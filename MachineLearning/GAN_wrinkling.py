


#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import sklearn as sk
import scipy as sc
import os
from os.path import join
from sklearn.metrics import mean_squared_error

import tensorflow_datasets as tfds

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

data_df = dd.read_hdf(join(path,'sample_set.hdf'),key='DNS')


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

col_names = data_df.columns


#%%
# scale the data
scaler_X = myStandardScaler()
scaler_y = myStandardScaler()

data_df=data_df[data_df['omega_model_planar']>1e-2]
data_df=data_df[data_df['c_bar']<0.99]
data_df=data_df[data_df['c_bar']>0.1].compute()        # sample from the entire data set

#%%
# compute tensors R, S mag(U) etc.

mag_U = np.sqrt(data_df['U_bar'].values**2 + data_df['V_bar'].values**2 +data_df['W_bar'].values**2)
mag_grad_c = np.sqrt(data_df['grad_c_x_LES'].values**2 + data_df['grad_c_y_LES'].values**2 +data_df['grad_c_z_LES'].values**2)

sum_U = data_df['U_bar'].values + data_df['V_bar']+data_df['W_bar'].values
sum_c = abs(data_df['grad_c_x_LES'].values) + abs(data_df['grad_c_y_LES'].values) +abs(data_df['grad_c_z_LES'].values)

grad_U = np.sqrt(data_df['grad_U_x_LES'].values**2 + data_df['grad_U_y_LES'].values**2 +data_df['grad_U_z_LES'].values**2)
grad_V = np.sqrt(data_df['grad_V_x_LES'].values**2 + data_df['grad_V_y_LES'].values**2 +data_df['grad_V_z_LES'].values**2)
grad_W = np.sqrt(data_df['grad_W_x_LES'].values**2 + data_df['grad_W_y_LES'].values**2 +data_df['grad_W_z_LES'].values**2)

mag_grad_U = np.sqrt(grad_U**2 + grad_V**2 +grad_W**2)
sum_grad_U = abs(grad_U) + abs(grad_V) +abs(grad_W)

gradient_tensor = np.array([
                    [data_df['grad_U_x_LES'].values,data_df['grad_V_x_LES'].values,data_df['grad_W_x_LES'].values],
                    [data_df['grad_U_y_LES'].values,data_df['grad_V_y_LES'].values,data_df['grad_W_y_LES'].values],
                    [data_df['grad_U_z_LES'].values,data_df['grad_V_z_LES'].values,data_df['grad_W_z_LES'].values],
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

data_df['mag_grad_c'] = mag_grad_c
data_df['mag_U'] = mag_U
data_df['sum_c'] = sum_c
data_df['sum_U'] = sum_U
data_df['sum_grad_U'] = sum_grad_U
data_df['mag_grad_U'] = mag_grad_U

data_df['lambda_1'] = lambda_1
data_df['lambda_2'] = lambda_2
data_df['lambda_3'] = lambda_3
data_df['lambda_4'] = lambda_4
data_df['lambda_5'] = lambda_5
#%%
X_data = data_df[['c_bar', 'omega_model_planar', 'mag_U', 'mag_grad_U','sum_U','sum_grad_U','mag_grad_c','sum_c',
                      'lambda_1','lambda_2','lambda_3','lambda_3','lambda_4','SGS_flux','lambda_5','UP_delta']] #[['c_bar', 'omega_model_planar', 'U_bar', 'V_bar', 'W_bar', 'U_prime', 'V_prime', 'W_prime']]
y_data = data_df['omega_DNS_filtered']

X_scaled = scaler_X.fit_transform(X_data.values)
y_scaled = scaler_y.fit_transform(y_data.values.reshape(-1,1))

# X_3D = X_scaled.reshape(512,512,512,X_data.shape[1])
# y_3D = y_scaled.reshape(512,512,512,1)

# pos = 50
#
# X_train = X_3D[:,:,:pos,:].reshape(pos*100*100,8)
# X_test = X_3D[:,:,pos:,:].reshape(100-pos*100*100,8)
#
# y_train = y_3D[:,:,:pos,:].reshape(pos*100*100,1)
# y_test = y_3D[:,:,pos:,:].reshape(100-pos*100*100,1)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,shuffle=True,test_size=0.1)

X_test,X_val,y_test,y_val = train_test_split(X_test,y_test,shuffle=True,test_size=0.3)

#%%
# create Tensorflow datasets from python files

dataset_train = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X_train,dtype=tf.float32), tf.convert_to_tensor(y_train,dtype=tf.float32))).batch(batch_size=100)
dataset_validation = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X_val,dtype=tf.float32), tf.convert_to_tensor(y_val,dtype=tf.float32))).batch(batch_size=100)
dataset_test = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X_test,dtype=tf.float32), tf.convert_to_tensor(y_test,dtype=tf.float32))).batch(batch_size=100)


#%%
''' Set up the generator model'''

nr_features=X_train.shape[1]

def make_generator_model():
    model = Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(nr_features, activation='linear')
    ])

    return model

'''Set up the discriminator'''


def make_discriminator_model():
    model = Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    return model

#%%
generator = make_generator_model()

noise = tf.random.normal([1,1 ])
generated_features = generator(noise, training=False)

print(generated_features.shape)


#%%
discriminator = make_discriminator_model()
decision = discriminator(generated_features)
print(decision)

#%%
'''Define the loss and optimizers'''
# This method returns a helper function to compute mae loss
mae = tf.keras.losses.mae
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#%%

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


#%%

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


#%%
# set up a simple model

# ANN parameters
dim_input = X_train.shape[1]
dim_label = y_train.shape[1]

DNN = Sequential([
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
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
for i in range(3):
    a += base * clc ** (i)
print(a)
epochs, c_len = a, base
schedule = SGDRScheduler(min_lr=1e-6, max_lr=1e-4,
                         steps_per_epoch=np.ceil(epoch_size / batch_size),
                         cycle_length=c_len, lr_decay=0.6, mult_factor=clc)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<1e-4):
            print("\nReached loss < 1e-4 so cancelling training!")
            self.model.stop_training = True

loss_callback = myCallback()
earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6)

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             save_freq=100)

callbacks = [schedule,loss_callback,earlyStop_callback]
#%%
DNN.summary()

#%%
history = DNN.fit(dataset_train,
                  epochs=epochs,
                  #batch_size=batch_size,
                  #validation_split=0.1,
                  shuffle=True,
                  #callbacks=callbacks,
                  validation_data=dataset_validation)

#%%

y_pred = DNN.predict(dataset_test)

y_pred_rescale = scaler_y.rescale(y_pred)
y_test_rescale = scaler_y.rescale(y_test)

X_test_rescale = scaler_X.rescale(X_test)

#%%
plt.figure()
plt.scatter(y_pred_rescale,y_test_rescale,s=0.2)

plt.figure()
plt.scatter(X_test_rescale[:,0],y_test_rescale[:],c='b',s=0.2)
plt.scatter(X_test_rescale[:,0],y_pred_rescale[:],c='r',s=0.2)
plt.scatter(X_test_rescale[:,0],X_test_rescale[:,1],c='k',s=0.2)
plt.legend(['Test data','Prediction','Pfitzner model'])
plt.xlabel('c_bar')
plt.ylabel('omega')


plt.figure()
plt.semilogy(history.history['loss'],scaley='log')
plt.semilogy(history.history['val_loss'],scaley='log')
