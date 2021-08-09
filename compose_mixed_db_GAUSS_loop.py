#join the single parquet files to one large DB for the training

# this works: Oct 2020

import numpy as np
import dask.dataframe as dd
import pandas as pd
from os.path import join
import dask
import dask.array as da
import matplotlib.pyplot as plt
import argparse
import os

def compose_db(case1,case2,train_test_set,scaler):
    '''

    :param case: UPRIMEXY case
    :param train_test_set: test or train set
    :param scaler: Standard, or Log and Standard scaler
    :return:
    '''

    path_to_data1 = case1+'/postProcess_DNN/'
    path_to_data2 = case2+'/postProcess_DNN/'

    out_path = 'UPRIME_5_15_mix'

    # get the relevant files in path_to_data_1
    filenames_1 = os.listdir(path_to_data1)
    filenames_1 = [f for f in filenames_1 if f.endswith(train_test_set+'.parquet')]

    # get the relevant files in path_to_data_2
    filenames_2 = os.listdir(path_to_data2)
    filenames_2 = [f for f in filenames_2 if f.endswith(train_test_set+'.parquet')]

    # no of output files

    if train_test_set=='test':
        files = 1
        FOLDER='TEST'

        # read in the data as dask dataframe
        data_pq_1 = dd.read_parquet(
            join(path_to_data1, 'filter_width_*_DNN_' + case1 + '_' + train_test_set + '.parquet'))
        data_pq_2 = dd.read_parquet(
            join(path_to_data2, 'filter_width_*_DNN_' + case2 + '_' + train_test_set + '.parquet'))
        # data_pq = dd.read_parquet(join(path_to_data,'train_example_*.parquet'))

        data_pq = dd.concat([data_pq_1, data_pq_2], ignore_index=True).compute()
        print(data_pq.describe())

    else:
        files = 10
        FOLDER='TRAIN'

        for f in filenames_1:
            if f == filenames_1[0]:
                data_pq_1 = pd.read_parquet(join(path_to_data1, f))
            else:
                data_pq_1 = pd.concat([data_pq_1, pd.read_parquet(join(path_to_data1, f))])

        print('files_1 done')

        for f in filenames_2:
            if f == filenames_2[0]:
                data_pq_2 = pd.read_parquet(join(path_to_data2, f)).sample(frac=0.5)
            else:
                data_pq_2 = pd.concat([data_pq_2, pd.read_parquet(join(path_to_data2, f)).sample(frac=0.5)])

        print('files_2 done')

        data_pq = pd.concat([data_pq_1,data_pq_2],ignore_index=True)
        print('concat done')
        del data_pq_1, data_pq_2
        data_pq= data_pq.sample(frac=1)
        print('reset index')
        data_pq = data_pq.reset_index(drop=True)
        print(data_pq.describe())

    # check if Log scaler
    if scaler=='Log':
        print('\nApplying Log Transformation to the data set')
        data_pq = data_pq.apply(np.log,axis=1)

    # get the mean and STD of the dataset
    data_mean = data_pq.mean()
    data_std = data_pq.std()

    #assign variables to dataFrame
    moments = pd.DataFrame(data_mean,columns=['mean'])
    del data_mean
    print('mean done')
    moments['std'] = data_std
    del data_std
    print('std done')

    # write down moments file
    if train_test_set=='train':
        if scaler=='Log':
            print('\nComputing moments for Log transformed...')
            moments.to_csv(join(out_path, 'moments_mix_Log.csv'))
      
            len_data = len(data_pq)
            chunk_size = int(len_data/files)-1

        elif scaler == 'Standard':
            print('\nComputing moments for Standard Scaler...')
            moments.to_csv(join(out_path, 'moments_mix.csv'))

            len_data = len(data_pq)
            chunk_size = len_data #int(len_data/files)-1

        print('Moments are written.')
    else:
        print('\nNo moments computed for test set ...')

    #len_data = len(data_pq)
    #chunk_size = int(len_data/files)-1

    for i in range(1,files+1):

        #data_pq = data_pq.repartition(npartitions=1)
        #print(i)
        #sample and change format
        data_df = data_pq.iloc[(i-1)*chunk_size:(i)*chunk_size].astype(np.float32)#.sample(frac=1/files)
        #print(data_df.head())
        
        # sample again to loose dask indexing
        #data_df = data_df.sample(frac=1.0).astype(np.float32)#.drop('Unnamed: 0',axis=1)

        # write the data base
        filename=join(out_path,FOLDER,train_test_set+'_'+case1+case2+'_'+str(i))

        ## PARQUET
        if scaler=='Log':
            data_df.to_parquet(filename+'_Log.parquet')
        elif scaler=='Standard':
            data_df.to_parquet(filename + '.parquet')

    #plt.show()

    print('\nAll done. Bye.\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="case: UPRIME5, UPRIME75 .., train_test_set: train or test")
    parser.add_argument('--train_test_set',type=str)
    parser.add_argument('--case1',type=str)
    parser.add_argument('--case2',type=str)
    parser.add_argument('--scaler', type=str) #TODO: Standard, Log, maybe more...
    args = parser.parse_args()

    # run the function
    compose_db(args.case1,args.case2,args.train_test_set,args.scaler)
