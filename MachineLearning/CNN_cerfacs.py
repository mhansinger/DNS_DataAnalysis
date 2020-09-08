import tensorflow as tf
from tensorflow.keras import backend as K

# from tensorflow.keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
# K.clear_session()
import numpy as np
import h5py
import random
from scipy.stats import norm
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers, Input, optimizers, layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import History, ModelCheckpoint
import os
from glob import glob

nb_epochs = 300
steps_train = 100
steps_valid = 50
size_block = 16  # (16*16*16)
initial_LR = 0.01
Decay = 0.8  # Decay applied every 10epochs
adim_8 = 342.553

if not os.path.exists('save_model'):
    os.makedirs('save_model')


def pad_fields(passive, target, adim, size_block_x, size_block_yz):
    padded_passive = np.pad(passive, (
    (0, size_block_x), (size_block_yz // 2, size_block_yz // 2), (size_block_yz // 2, size_block_yz // 2)), mode="edge")
    # periodicity y
    padded_passive[:, :size_block_yz // 2] = padded_passive[:, -2 * (size_block_yz // 2): -(size_block_yz // 2)]
    padded_passive[:, -(size_block_yz // 2):] = padded_passive[:, (size_block_yz // 2): 2 * (size_block_yz // 2)]
    # periodicity z
    padded_passive[:, :, :size_block_yz // 2] = padded_passive[:, :, -2 * (size_block_yz // 2): -(size_block_yz // 2)]
    padded_passive[:, :, -(size_block_yz // 2):] = padded_passive[:, :, (size_block_yz // 2): 2 * (size_block_yz // 2)]

    target = target / adim
    padded_target = np.pad(target, (
    (0, size_block_x), (size_block_yz // 2, size_block_yz // 2), (size_block_yz // 2, size_block_yz // 2)), mode="edge")
    # periodicity y
    padded_target[:, :size_block_yz // 2] = padded_target[:, -2 * (size_block_yz // 2): -(size_block_yz // 2)]
    padded_target[:, -(size_block_yz // 2):] = padded_target[:, (size_block_yz // 2): 2 * (size_block_yz // 2)]
    # periodicity z
    padded_target[:, :, :size_block_yz // 2] = padded_target[:, :, -2 * (size_block_yz // 2): -(size_block_yz // 2)]
    padded_target[:, :, -(size_block_yz // 2):] = padded_target[:, :, (size_block_yz // 2): 2 * (size_block_yz // 2)]
    return (padded_passive, padded_target)


def generator_3D(begin_distrib, end_distrib, size_num, rotate, flip):  # valid_or_train= 1 for validation, 2 for train
    while 1:

        crops_per_block = 10

        batch_size = size_num * crops_per_block
        liste_input = np.zeros((batch_size, size_block, size_block, size_block, 1))
        liste_target = np.zeros((batch_size, size_block, size_block, size_block, 1))

        for num in range(size_num):

            num_DNS = np.random.randint(1, 3)  # DNS 1 or 2
            liste_name = sorted(glob("DATA/DNS" + str(num_DNS) + "*.h5"))
            choosen_num = np.random.randint(begin_distrib, end_distrib)
            my_file = h5py.File(liste_name[choosen_num], 'r')
            passive = my_file['filt_8'].value
            target = my_file['filt_grad_8'].value
            padded_passive, padded_target = pad_fields(passive, target, adim_8, size_block, size_block)
            for k in range(crops_per_block):
                abs_x = np.random.randint(0, 64 - size_block + 1)
                abs_y = np.random.randint(0, 32)  # with padding
                abs_z = np.random.randint(0, 32)
                liste_input[num * crops_per_block + k, :, :, :, 0] = padded_passive[abs_x:abs_x + size_block,
                                                                     abs_y:abs_y + size_block, abs_z:abs_z + size_block]
                liste_target[num * crops_per_block + k, :, :, :, 0] = padded_target[abs_x:abs_x + size_block,
                                                                      abs_y:abs_y + size_block,
                                                                      abs_z:abs_z + size_block]

        if rotate:
            rot_times = np.random.randint(4, size=batch_size)
            rot_ax = np.random.randint(3, size=batch_size)
            ax_to_plane = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
            for i, (ti, ax) in enumerate(zip(rot_times, rot_ax)):
                liste_input[i,] = np.rot90(liste_input[i,], k=ti, axes=ax_to_plane[ax])
                liste_target[i,] = np.rot90(liste_target[i,], k=ti, axes=ax_to_plane[ax])

        if flip:
            for i, flip_by in enumerate(np.random.randint(6, size=batch_size)):
                if flip_by <= 2:
                    liste_input[i] = np.flip(liste_input[i], axis=flip_by)
                    liste_target[i] = np.flip(liste_target[i], axis=flip_by)

        indice = np.arange(liste_input.shape[0])
        np.random.shuffle(indice)
        liste_input = liste_input[indice]
        liste_target = liste_target[indice]

        yield liste_input, liste_target


def CNN():
    num_channels = 1
    num_mask_channels = 1
    img_shape = (None, None, None, 1)

    inputs = Input(shape=img_shape)
    conv1 = layers.Conv3D(32, 3, padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = layers.Conv3D(32, 3, padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = layers.Conv3D(64, 3, padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = layers.Conv3D(64, 3, padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = layers.Conv3D(128, 3, padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = layers.Conv3D(128, 3, padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = layers.UpSampling3D(size=(2, 2, 2))(conv3)

    up4 = layers.concatenate([conv3, conv2])
    conv4 = layers.Conv3DTranspose(64, 3, padding='same')(up4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = layers.Conv3DTranspose(64, 3, padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)  ##conv ou crop
    conv4 = Activation('relu')(conv4)
    conv4 = layers.Conv3DTranspose(64, 1, padding='same')(conv4)
    conv4 = layers.UpSampling3D(size=(2, 2, 2))(conv4)

    up5 = layers.concatenate([conv4, conv1])
    conv5 = layers.Conv3DTranspose(32, 3, padding='same')(up5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = layers.Conv3DTranspose(32, 3, padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)  ##conv ou crop
    conv5 = Activation('relu')(conv5)
    conv5 = layers.Conv3DTranspose(1, 1, padding='same', activation='relu')(conv5)

    model = Model(inputs=inputs, outputs=conv5)
    # model.summary()

    return (model)


def scheduler(epoch):
    initial_lrate = K.get_value(model.optimizer.lr)
    if epoch % 10 == 9:
        lrate = initial_lrate * Decay
        return lrate
    return initial_lrate


# if __name__ == "__main__":
#     train_generator = generator_3D(1, 41, 4, True, True)  # first 40 fields for training cf paper
#     valid_generator = generator_3D(41, 51, 1, False, False)  # Last 10 for validation
#
#     model = CNN()
#     new_Adam = optimizers.Adam(lr=initial_LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#     model.compile(optimizer=new_Adam, loss='mse', metrics=['mae'])
#
#     lrate = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
#
#     filepath = "save_model/unet_{epoch:03d}_with_loss_{val_loss:.4f}.h5"
#     checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
#
#     history = model.fit_generator(train_generator, steps_per_epoch=steps_train, epochs=nb_epochs,
#                                   validation_data=valid_generator, callbacks=[lrate, checkpoint],
#                                   validation_steps=steps_valid)
