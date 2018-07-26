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
from numba import jit


class data_binning(object):
    def __init__(self,case='1bar',Nx=250):

        self.c_path = join(case,'rho_by_c.dat')
        self.rho_path = join(case,'rho.dat')
        self.data_c = None
        self.data_rho = None
        self.Nx = Nx
        self.case = case

        # checks if output directory exists
        self.output_path = join(case,'output')
        if os.path.isdir(self.output_path) is False:
            os.mkdir(self.output_path)

        self.col_names = ['c_tilde','rho_bar','c_rho','rho','x','y','z']

    def dask_read_transform(self):

        try:
            self.data_c = dd.read_csv(self.c_path,names=['rho_c'])
        except:
            print('No data for C_rho')

        try:
            self.data_rho = dd.read_csv(self.rho_path,names = ['rho'])
        except:
            print('No data for rho')

        # transform the data into an array and reshape
        self.rho_data_da = da.asarray(self.data_rho).reshape(self.Nx,self.Nx,self.Nx)
        self.c_data_da = da.asarray(self.data_c).reshape(self.Nx,self.Nx,self.Nx)

    @jit
    def run_analysis(self,filter_width = 8, interval = 2, threshold=0.05, c_rho_max = 0.1818,histogram=True ):
        self.filter_width  =filter_width
        self.threshold = threshold
        self.interval = interval
        self.c_rho_max = c_rho_max

        count = 0
        for k in range(self.filter_width-1,self.Nx,self.interval):
            for j in range(self.filter_width - 1, self.Nx, self.interval):
                for i in range(self.filter_width - 1, self.Nx, self.interval):

                    # TEST VERSION
                    this_set = self.c_data_da[i-self.filter_width:i ,j-self.filter_width:j, k-self.filter_width:k].compute()

                    # check if threshold condition is reached
                    if (this_set > self.threshold).any() and (this_set < self.c_rho_max).all():

                        #print('True! ',this_set.mean())

                        #compute c-bar
                        self.compute_cbar(this_set,i,j,k,histogram)

    @jit
    def compute_cbar(self,data_set,i,j,k,histogram):

        this_rho_c_reshape = data_set.reshape(self.filter_width**3)
        this_rho_c_mean = this_rho_c_reshape.mean()

        # get the density for the relevant points! it is stored in a different file!
        this_rho_reshape = self.rho_data_da[i-self.filter_width:i ,j-self.filter_width:j, k-self.filter_width:k].compute().reshape(self.filter_width**3)

        this_rho_mean = this_rho_reshape.mean()

        # compute c_tilde: mean(rho*c)/mean(rho)
        c_tilde = this_rho_c_mean / this_rho_mean

        # another criteria
        if c_tilde < 0.99:
            # construct empty data array and fill it
            data_arr = np.zeros((self.filter_width ** 3, len(self.col_names)))
            data_arr[:, 0] = c_tilde
            data_arr[:, 1] = this_rho_mean
            data_arr[:, 2] = this_rho_c_reshape
            data_arr[:, 3] = this_rho_reshape
            data_arr[:, 4] = int(i)
            data_arr[:, 5] = int(j)
            data_arr[:, 6] = int(k)
            #
            data_df = pd.DataFrame(data_arr, columns=self.col_names)
            file_name = 'c_tilde_%.5f_filter_%i_%s.csv' % (c_tilde, self.filter_width, self.case)
            data_df.to_csv(join(self.output_path, file_name), index=False)

            if histogram:
                self.plot_histograms(c_tilde=c_tilde,this_rho_c_reshape=this_rho_c_reshape,this_rho_reshape=this_rho_reshape)
                #self.plot_histograms()
            #print('c_tilde: ', c_tilde)

    def plot_histograms(self,c_tilde,this_rho_c_reshape,this_rho_reshape):#(self, c_tilde, this_rho_c_reshape, this_rho_reshape):
        # plot the c_tilde and c, both normalized

        n_bins = 20 #int(self.filter_width**3 / 10)

        c_max = 0.182363

        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
        ax0.hist(this_rho_c_reshape,bins=n_bins,normed=True)
        ax0.set_title('rho x c')
        ax0.set_xlim(0,c_max)

        ax1.hist(this_rho_c_reshape/this_rho_reshape,bins=n_bins,normed=True)
        ax1.set_title('c')
        ax1.set_xlim(0,1)

        fig.tight_layout()
        plt.suptitle('c_tilde = %.3f' % c_tilde)
        fig_name = join(self.output_path,'c_tilde_%.5f_filter_%i_%s.png' % (c_tilde, self.filter_width, self.case))
        plt.savefig(fig_name)

        print('c_tilde: ', c_tilde)
        print(' ')

        # plt.figure()
        # plt.title('c_tilde = %.5f' % c_tilde)
        # plt.subplot(211)
        # plt.hist(this_rho_c_reshape,bins=n_bins,normed=True)
        # plt.title('rho x c')
        #
        # plt.subplot(221)
        # plt.hist(this_rho_c_reshape/this_rho_reshape,bins=n_bins,normed=True)
        # plt.title('c')


if __name__ == "__main__":

    print('Starting 1bar case!')

    bar1 =data_binning()
    bar1.dask_read_transform()
    bar1.run_analysis(filter_width=8,interval=8,histogram=True)





