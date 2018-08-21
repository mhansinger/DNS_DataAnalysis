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
    def __init__(self,case='1bar',Nx=250,alpha=0.81818,beta=6,befact=7361):

        self.c_path = join(case,'rho_by_c.dat')
        self.rho_path = join(case,'rho.dat')
        self.data_c = None
        self.data_rho = None
        self.Nx = Nx
        self.case = case

        # for reaction rates
        self.alpha =alpha
        self.beta = beta
        self.bfact = befact

        # checks if output directory exists
        self.output_path = join(case,'output')
        if os.path.isdir(self.output_path) is False:
            os.mkdir(self.output_path)

        self.col_names = ['c_tilde','rho_bar','c_rho','rho','reactionRate']

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
    def run_analysis_wrinkling(self,filter_width = 8, interval = 2, threshold=0.05, c_rho_max = 0.1818,histogram=True ):
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
                        self.compute_cbar_wrinkling(this_set,i,j,k,histogram)

    @jit
    def compute_cbar(self,data_set,i,j,k,histogram):

        this_rho_c_reshape = data_set.reshape(self.filter_width**3)
        this_rho_c_mean = this_rho_c_reshape.mean()

        # get the density for the relevant points! it is stored in a different file!
        this_rho_reshape = self.rho_data_da[i-self.filter_width:i ,j-self.filter_width:j, k-self.filter_width:k].compute().reshape(self.filter_width**3)

        # c without density
        this_c_reshape= this_rho_c_reshape/this_rho_reshape

        this_rho_mean = this_rho_reshape.mean()

        # compute c_tilde: mean(rho*c)/mean(rho)
        c_tilde = this_rho_c_mean / this_rho_mean

        # compute the reaction rate of each cell     Eq (2.28) Senga documentation
        # note that for Le=1: c=T, so this_c_reshape = this_T_reshape
        exponent = - self.beta*(1-this_c_reshape) / (1 - self.alpha*(1 - this_c_reshape))
        this_RR_reshape = self.bfact*this_rho_reshape*(1-this_c_reshape)*np.exp(exponent)

        # another criteria
        if c_tilde < 0.99:
            # construct empty data array and fill it
            data_arr = np.zeros((self.filter_width ** 3, len(self.col_names)))
            data_arr[:, 0] = c_tilde
            data_arr[:, 1] = this_rho_mean
            data_arr[:, 2] = this_rho_c_reshape
            data_arr[:, 3] = this_rho_reshape
            data_arr[:, 4] = this_RR_reshape
            #data_arr[:, 5] = int(i)
            #data_arr[:, 6] = int(j)
            #data_arr[:, 7] = int(k)
            #
            data_df = pd.DataFrame(data_arr, columns=self.col_names)
            file_name = 'c_tilde_%.5f_filter_%i_%s.csv' % (c_tilde, self.filter_width, self.case)
            data_df.to_csv(join(self.output_path, file_name), index=False)

            if histogram:
                self.plot_histograms(c_tilde=c_tilde,this_rho_c_reshape=this_rho_c_reshape,this_rho_reshape=this_rho_reshape,
                                     this_RR_reshape=this_RR_reshape)
                #self.plot_histograms()
            #print('c_tilde: ', c_tilde)

    def plot_histograms(self,c_tilde,this_rho_c_reshape,this_rho_reshape,c_bar,this_RR_reshape,wrinkling=1):
        # plot the c_tilde and c, both normalized

        n_bins = 20 #int(self.filter_width**3 / 10)

        c_max = 0.182363

        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
        ax0.hist(this_rho_c_reshape,bins=n_bins,normed=True,edgecolor='black', linewidth=0.7)
        ax0.set_title('rho*c')
        ax0.set_xlim(0,c_max)

        this_c_reshape = this_rho_c_reshape/this_rho_reshape
        #ax1.hist(this_c_reshape,bins=n_bins,normed=True,edgecolor='black', linewidth=0.7)
        ax1.set_title('c')
        ax1.set_xlim(0,1)

        ax2 = ax1.twinx()
        color = 'r'
        ax2.set_ylabel('Reaction Rate', color=color)
        # ax2.scatter(this_c_reshape, this_RR_reshape,color=color,s=0.9)
        #ax1.hist(this_RR_reshape/this_RR_reshape.max(), bins=n_bins,normed=True,color=color,alpha=0.5,edgecolor='black', linewidth=0.7)
        ax1.hist(this_c_reshape, bins=n_bins, normed=True, edgecolor='black', linewidth=0.7)
        ax2.scatter(this_c_reshape, this_RR_reshape, color=color, s=0.9)
        ax2.set_ylim(0,90)

        #fig.tight_layout()
        plt.suptitle('c_tilde = %.3f; c_bar = %.3f; wrinkling = %.3f; RR_mean = %.2f\n' % (c_tilde,c_bar,wrinkling,this_RR_reshape.mean()))
        fig_name = join(self.output_path,'c_tilde_%.4f_wrinkl_%.3f_filter_%i_%s.png' % (c_tilde, wrinkling,self.filter_width, self.case))
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

    @jit
    def compute_cbar_wrinkling(self,data_set,i,j,k,histogram):
        # this is to compute with the wrinkling factor

        this_rho_c_reshape = data_set.reshape(self.filter_width**3)

        this_rho_c_mean = this_rho_c_reshape.mean()

        # get the density for the relevant points! it is stored in a different file!
        this_rho_reshape = self.rho_data_da[i-self.filter_width:i ,j-self.filter_width:j, k-self.filter_width:k].compute().reshape(self.filter_width**3)

        # c without density
        this_c_reshape = this_rho_c_reshape / this_rho_reshape

        this_c_bar = this_c_reshape.mean()

        this_rho_mean = this_rho_reshape.mean()

        # compute c_tilde: mean(rho*c)/mean(rho)
        c_tilde = this_rho_c_mean / this_rho_mean

        # compute the reaction rate of each cell     Eq (2.28) Senga documentation
        # note that for Le=1: c=T, so this_c_reshape = this_T_reshape
        exponent = - self.beta*(1-this_c_reshape) / (1 - self.alpha*(1 - this_c_reshape))
        this_RR_reshape = self.bfact*this_rho_reshape*(1-this_c_reshape)*np.exp(exponent)

        # another criteria
        if 0.7 < this_c_bar < 0.9:
            #print(this_c_reshape)
            print("c_bar: ",this_c_bar)
            this_wrinkling = self.get_wrinkling(this_c_bar,i,j,k)

            # consistency check, wrinkling factor needs to be >1!
            if this_wrinkling < 1:
                print("##################")
                print("Computed Nonsense!")
                print("##################")
            else:
                # construct empty data array and fill it
                data_arr = np.zeros((self.filter_width ** 3, len(self.col_names)))
                data_arr[:, 0] = c_tilde
                data_arr[:, 1] = this_rho_mean
                data_arr[:, 2] = this_rho_c_reshape
                data_arr[:, 3] = this_rho_reshape
                data_arr[:, 4] = this_RR_reshape
                # data_arr[:, 5] = int(i)
                # data_arr[:, 6] = int(j)
                # data_arr[:, 7] = int(k)
                #
                data_df = pd.DataFrame(data_arr, columns=self.col_names)
                file_name = 'c_tilde_%.4f_wrinkl_%.3f_filter_%i_%s.csv' % (
                c_tilde, this_wrinkling, self.filter_width, self.case)
                data_df.to_csv(join(self.output_path, file_name), index=False)

                if histogram:
                    self.plot_histograms(c_tilde=c_tilde, this_rho_c_reshape=this_rho_c_reshape,
                                         this_rho_reshape=this_rho_reshape, c_bar=this_c_bar,
                                         this_RR_reshape=this_RR_reshape, wrinkling=this_wrinkling)
                    # self.plot_histograms()
                # print('c_tilde: ', c_tilde)

    @jit
    def get_wrinkling(self,this_cbar,i,j,k):
        # computes the wriknling factor from resolved and filtered flame surface
        #print(i)
        this_A_LES = self.get_A_LES(this_cbar,i,j,k)

        this_A_DNS = self.get_A_DNS(i,j,k)

        print("Wrinkling factor: ", this_A_DNS/this_A_LES)
        print(" ")

        return this_A_DNS/this_A_LES

    @jit
    def get_A_DNS(self,i,j,k):
        # initialize fields
        #print('I m in!')
        width = 1

        this_DNS_gradX = np.zeros((self.filter_width,self.filter_width,self.filter_width))
        this_DNS_gradY = this_DNS_gradX.copy()
        this_DNS_gradZ = this_DNS_gradX.copy()
        this_DNS_magGrad_c = this_DNS_gradX.copy()

        this_rho_c_data = self.c_data_da[i -(self.filter_width+1):(i + 1), (j - 1) - self.filter_width:(j + 1),
                      k - (self.filter_width+1):(k + 1)].compute()

        this_rho_data = self.rho_data_da[(i - 1) - self.filter_width:(i + 1), (j - 1) - self.filter_width:(j + 1),
                      k - (self.filter_width+1):(k + 1)].compute()

        #print("this rho_c_data.max(): ", (this_rho_c_data/this_rho_data))

        for l in range(self.filter_width):
            for m in range(self.filter_width):
                for n in range(self.filter_width):
                    this_DNS_gradX[l, m, n] = (this_rho_c_data[l+1, m, n]/this_rho_data[l+1, m, n] - this_rho_c_data[l-1,m, n]/this_rho_data[l-1,m, n])/(2 * width)
                    this_DNS_gradY[l, m, n] = (this_rho_c_data[l, m+1, n]/this_rho_data[l, m+1, n] - this_rho_c_data[l, m-1, n]/this_rho_data[l, m-1, n]) / (2 * width)
                    this_DNS_gradZ[l, m, n] = (this_rho_c_data[l, m, n+1]/this_rho_data[l, m, n+1] - this_rho_c_data[l, m, n-1]/this_rho_data[l, m, n-1]) / (2 * width)
                    # compute the magnitude of the gradient
                    this_DNS_magGrad_c[l,m,n] = np.sqrt(this_DNS_gradX[l,m,n]**2 + this_DNS_gradY[l,m,n]**2 + this_DNS_gradZ[l,m,n]**2)
                    # print("this_DNS_magGrad_c[l,m,n]: ",this_DNS_magGrad_c[l,m,n])
                    # print("this_c_data[l, m, n]: ", this_rho_c_data[l, m, n])
                    # print("this_c_data[l, m, n]: ", this_rho_c_data[l, m, n])
                    # print("this_c_data[l, m, n]: ", this_rho_c_data[l, m, n])

        # return the resolved iso surface area
        print("this_DNS_magGrad.sum(): ", this_DNS_magGrad_c.sum())
        print("max thisDNS_grad_X: ", this_DNS_gradX.max())
        print("max thisDNS_grad_Y: ", this_DNS_gradY.max())
        print("max thisDNS_grad_Z: ", this_DNS_gradZ.max())
        print("A_DNS: ", this_DNS_magGrad_c.sum() / (self.filter_width**3))

        return this_DNS_magGrad_c.sum() / (self.filter_width**3)

    @jit
    def get_A_LES(self,this_cbar,i,j,k):
        # computes the filtered iso surface

        if i - self.filter_width < 0 or j - self.filter_width < 0 or k - self.filter_width < 0:
            print("too small!")
            return 1000

        elif i + self.filter_width > self.Nx or j + self.filter_width > self.Nx or k + self.filter_width > self.Nx:
            print("too big!")
            return 1000

        else:
            # get the neighbour rho data
            this_rho_west = self.rho_data_da[i - self.filter_width:i, j - 2 * self.filter_width:j - self.filter_width,
                            k - self.filter_width:k].compute()
            this_rho_east = self.rho_data_da[i - self.filter_width:i, j:j + self.filter_width,
                            k - self.filter_width:k].compute()

            this_rho_north = self.rho_data_da[i:i + self.filter_width, j - self.filter_width:j,
                             k - self.filter_width:k].compute()
            this_rho_south = self.rho_data_da[i - 2 * self.filter_width:i - self.filter_width, j - self.filter_width:j,
                             k - self.filter_width:k].compute()

            this_rho_up = self.rho_data_da[i - self.filter_width:i, j - self.filter_width:j,
                          k:k + self.filter_width].compute()
            this_rho_down = self.rho_data_da[i - self.filter_width:i, j - self.filter_width:j,
                            k - 2 * self.filter_width:k - self.filter_width].compute()

            # get the neighbour c data
            this_c_west = self.c_data_da[i - self.filter_width:i, j - 2 * self.filter_width:j - self.filter_width,
                          k - self.filter_width:k].compute() / this_rho_west
            this_c_east = self.c_data_da[i - self.filter_width:i, j:j + self.filter_width,
                          k - self.filter_width:k].compute() / this_rho_east

            this_c_north = self.c_data_da[i:i + self.filter_width, j - self.filter_width:j,
                           k - self.filter_width:k].compute() / this_rho_north
            this_c_south = self.c_data_da[i - 2 * self.filter_width:i - self.filter_width, j - self.filter_width:j,
                           k - self.filter_width:k].compute() / this_rho_south

            this_c_up = self.c_data_da[i - self.filter_width:i, j - self.filter_width:j,
                        k:k + self.filter_width].compute() / this_rho_up
            this_c_down = self.c_data_da[i - self.filter_width:i, j - self.filter_width:j,
                          k - 2 * self.filter_width:k - self.filter_width].compute() / this_rho_down

            # now computing the gradients
            this_grad_X = (this_c_north.mean() - this_c_south.mean()) / (2 * self.filter_width)
            this_grad_Y = (this_c_east.mean() - this_c_west.mean()) / (2 * self.filter_width)
            this_grad_Z = (this_c_down.mean() - this_c_up.mean()) / (2 * self.filter_width)

            this_magGrad_c = np.sqrt(this_grad_X ** 2 + this_grad_Y ** 2 + this_grad_Z ** 2)

            # print('i - 2*%f' % self.filter_width)
            print('this_c_north mean %f' % this_c_north.mean())
            print('this_c_south mean %f' % this_c_south.mean())
            print('this_c_west mean %f' % this_c_west.mean())
            print('this_c_east mean %f' % this_c_east.mean())
            print('this_c_down mean %f' % this_c_down.mean())
            print('this_c_up mean %f' % this_c_up.mean())
            print('this_grad_X', this_grad_X)
            print('this_grad_Y', this_grad_Y)
            print('this_grad_Z', this_grad_Z)
            print('A_LES: ', this_magGrad_c)

            # print("A_LES: ", this_magGrad)
            return this_magGrad_c


if __name__ == "__main__":

    print('Starting 1bar case!')

    bar1 =data_binning()
    bar1.dask_read_transform()
    bar1.run_analysis_wrinkling(filter_width=16,interval=8,histogram=True)





