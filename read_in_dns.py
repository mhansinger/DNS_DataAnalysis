'''
This is to read in the binary data File for the high pressure bunsen data

@author: mhansinger
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import dask.dataframe as dd
import dask.array as da


#############################

# pfad beschreibung

def read_in_dns(case='1bar',NX=250,base_path = '/home/max/HDD2_Data/DNS_bunsen/'):


    file_name = 'dmpi_merged.dat'

    cbar_name = 'plot_cbar.dat'
    file_path = base_path + case + '/' + file_name

    c_bar_path = base_path + case + '/' + cbar_name

    npoints = NX # depends on 1,5, 10 bar case


    #############################
    # read in the data as numpy array

    data_np = np.fromfile(file_path, dtype=np.float16)

    print('The data contains %f entries' % data_np.shape[0])
    print(' ')

    # reshape for the nfields

    length = npoints ** 3

    data_reshape = data_np[:-1].reshape(length, nfields)
    print(data_reshape.shape)


