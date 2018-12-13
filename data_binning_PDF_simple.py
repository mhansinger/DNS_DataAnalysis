'''
This is to read in the binary data File for the high pressure bunsen data

@author: mhansinger

last change: 13.12.2018
'''

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os
from os.path import join
import dask.dataframe as dd
import dask.array as da
from numba import jit
from mayavi import mlab
# to free memory
import gc



class data_binning_PDF(object):

    def __init__(self, case, m, alpha, beta, bins):
        '''
        # CONSTRUCTOR
        :param case:    case name: 1bar, 5bar or 10bar
        :param alpha:   FROM REACTION RATE
        :param beta:    FROM REACTION RATE
        :param m:       PARAMETER FOR PFITZNER PDF
        :param bins:    NUMBER OF HISTOGRAM BINS
        '''

        # THIS NAME IS GIVEN BY THE OUTPUT FROM FORTRAN CODE, COULD CHANGE...
        self.c_path = join(case,'rho_by_c.dat')
        self.rho_path = join(case,'rho.dat')

        self.data_c = None
        self.data_rho = None
        self.case = case
        self.bins = bins
        self.write_csv = False

        # Filter width of the LES cell: is filled later
        self.filter_width = None

        if self.case=='1bar':
            # NUMBER OF DNS CELLS IN X,Y,Z DIRECTION
            self.Nx = 250
            # PARAMETER FOR REACTION RATE
            self.bfact = 7364.0
            # REYNOLDS NUMBER
            self.Re = 100
            # DIMENSIONLESS DNS GRID SPACING, DOMAIN IS NOT UNITY
            self.delta_x = 1/188
            # PRESSURE [BAR]
            self.p = 1
        elif self.case=='5bar':
            self.Nx = 560
            self.bfact = 7128.3
            self.Re = 500
            self.delta_x = 1/432
            self.p = 5
        elif self.case=='10bar':
            self.Nx = 795
            self.bfact = 7128.3
            self.Re = 1000
            self.delta_x = 1/611
            self.p = 10
        elif self.case=='dummy_planar_flame':
            # this is a dummy case with 50x50x50 entries!
            print('\n################\nThis is the dummy test case!\n################\n')
            self.Nx = 150
            self.bfact = 7364.0
            self.Re = 100
            self.delta_x = 1/188
            self.p = 1
        else:
            raise ValueError('This case does not exist!\nOnly: 1bar, 5bar, 10bar\n')

        # for reaction rates
        self.alpha = alpha
        self.beta = beta
        self.m = m

        # normalizing pressure
        self.p_0 = 1

        # TO BE COMPUTED
        self._c_bar = None

        self._c_0 = None
        self.c_plus = None
        self.c_minus = None

        # SCHMIDT NUMBER
        self.Sc = 0.7

        # DELTA: NORMALIZED FILTERWIDTH
        self.Delta = None # --> is computed in self.run_analysis !

        # checks if output directory exists
        self.output_path = join(case,'output_test')
        if os.path.isdir(self.output_path) is False:
            os.mkdir(self.output_path)

        self.col_names = ['c_tilde','rho_bar','c_rho','rho','reactionRate']

        print('Case: %s' % self.case)
        print('Nr. of grid points: %i' % self.Nx)
        print("Re = %f" % self.Re)
        print("Sc = %f" % self.Sc)

        # CONSTRUCTOR END
        ########################

    def dask_read_transform(self):

        try:
            self.data_rho_c = dd.read_csv(self.c_path,names=['rho_c'])
        except:
            print('No data for C_rho')

        try:
            self.data_rho = dd.read_csv(self.rho_path,names = ['rho'])
        except:
            print('No data for rho')

        # transform the data into an array and reshape
        self.rho_data_da = da.asarray(self.data_rho).reshape(self.Nx,self.Nx,self.Nx)
        self.rho_c_data_da = da.asarray(self.data_rho_c).reshape(self.Nx,self.Nx,self.Nx)

    @jit
    def run_analysis(self,filter_width = 8, interval = 2, threshold=0.005, c_rho_max = 0.1818, histogram=True):
        # run the analysis without computation of wrinkling factor -> planar flame (dummy case)

        self.filter_width =filter_width
        self.threshold = threshold
        self.interval = interval
        self.c_rho_max = c_rho_max

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta = self.Sc * self.Re / np.sqrt(self.p/self.p_0) * self.delta_x * self.filter_width

        count = 0
        for k in range(self.filter_width-1,self.Nx,self.interval):
            for j in range(self.filter_width - 1, self.Nx, self.interval):
                for i in range(self.filter_width - 1, self.Nx, self.interval):

                    # TEST VERSION
                    # this is the current data cube which constitutes the LES cell
                    this_rho_c_set = self.rho_c_data_da[i-self.filter_width:i ,j-self.filter_width:j, k-self.filter_width:k].compute()

                    #print(this_rho_c_set)

                    # check if threshold condition is reached
                    # -> avoid computations where c_bar is either 0 or 1 as there is no flame front
                    if (this_rho_c_set > self.threshold).any() and (this_rho_c_set < self.c_rho_max).all():

                        #print('If criteria erreicht!')
                        #compute c-bar

                        self.compute_cbar(this_rho_c_set,i,j,k,histogram)

    @jit
    def compute_cbar(self,data_set,i,j,k,histogram):
        #compute c_bar without wrinkling factor

        this_rho_c_reshape = data_set.reshape(self.filter_width**3)
        this_rho_c_mean = this_rho_c_reshape.mean()

        # get the density for the relevant points! it is stored in a different file!
        this_rho_reshape = self.rho_data_da[i-self.filter_width:i ,j-self.filter_width:j, k-self.filter_width:k].compute().reshape(self.filter_width**3)
        this_rho_mean = this_rho_reshape.mean()

        # c without density
        this_c_reshape = this_rho_c_reshape / this_rho_reshape

        # c_bar is computed
        this_c_bar = this_c_reshape.mean()

        # compute c_tilde: mean(rho*c)/mean(rho)
        c_tilde = this_rho_c_mean / this_rho_mean

        # compute the reaction rate of each cell     Eq (2.28) Senga documentation
        # note that for Le=1: c=T, so this_c_reshape = this_T_reshape
        exponent = - self.beta*(1-this_c_reshape) / (1 - self.alpha*(1 - this_c_reshape))
        this_RR_reshape_DNS = self.bfact*this_rho_reshape*(1-this_c_reshape)*np.exp(exponent)

        # another criteria
        if this_c_bar < 0.99:
            # construct empty data array and fill it
            data_arr = np.zeros((self.filter_width ** 3, len(self.col_names)))
            data_arr[:, 0] = c_tilde
            data_arr[:, 1] = this_rho_mean
            data_arr[:, 2] = this_rho_c_reshape
            data_arr[:, 3] = this_rho_reshape
            data_arr[:, 4] = this_RR_reshape_DNS
            #data_arr[:, 5] = int(i)
            #data_arr[:, 6] = int(j)
            #data_arr[:, 7] = int(k)
            #
            data_df = pd.DataFrame(data_arr, columns=self.col_names)
            file_name = 'c_tilde_%.5f_filter_%i_%s.csv' % (c_tilde, self.filter_width, self.case)

            #############################################
            # computation of the integration boundaries
            self.compute_c_minus(this_c_bar)
            self.compute_c_plus(this_c_bar)
            print('c_plus: ',self.c_plus)
            #############################################

            if self.write_csv:
                data_df.to_csv(join(self.output_path, file_name), index=False)

            if histogram:
                self.plot_histograms_intervals(c_tilde=c_tilde,this_rho_c_reshape=this_rho_c_reshape,
                                               this_rho_reshape=this_rho_reshape,c_bar=this_c_bar,this_RR_reshape_DNS=this_RR_reshape_DNS)


    def plot_histograms_intervals(self,c_tilde,this_rho_c_reshape,this_rho_reshape,c_bar,this_RR_reshape_DNS,wrinkling=1):
        # plot c, RR, and integration boundaries c_plus, c_minus

        #fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 4))

        # compute the mean reaction rate for the single bins
        sorted_rho_c_reshape = np.sort(this_rho_c_reshape)
        idx = np.unravel_index(np.argsort(this_rho_c_reshape, axis=None), this_rho_c_reshape.shape)
        sorted_RR_reshape_DNS = np.array([x for y, x in sorted(zip(this_rho_c_reshape,this_RR_reshape_DNS))])

        hist, bin_edges = np.histogram(this_rho_c_reshape/this_rho_reshape, bins=self.bins)

        try:
            # check if dimensions are correct
            assert hist.sum() == self.filter_width**3
        except:
            print('Check the filter width and histogram')

        probability_c = hist/hist.sum()

        RR_LES_mean = 0
        RR_interval = 0
        counter = 0

        for id, val in enumerate(hist):
            RR_interval = sorted_RR_reshape_DNS[counter:counter+val].mean()
            RR_LES_mean = RR_LES_mean + (RR_interval*probability_c[id])

            counter = counter + val

        # GARBAGE COLLECT FREE MEMORY
        gc.collect()
        #print('RR_LES_mean: ', RR_LES_mean)

        ###################################

        this_c_reshape = this_rho_c_reshape/this_rho_reshape
        #ax1.set_title('c')
        #ax1.set_xlim(0,1)

        #ax2 = ax1.twinx()
        #color = 'r'
        #ax2.set_ylabel('Reaction Rate', color=color)
        #ax1.hist(this_c_reshape, bins=self.bins, normed=True, edgecolor='black', linewidth=0.7)
        #ax2.scatter(this_c_reshape, this_RR_reshape_DNS, color=color, s=0.9)
        #ax2.set_ylim(0,90)

        #fig.tight_layout()
        #plt.suptitle('c_bar = %.3f;  RR_mean_DNS = %.2f; c_plus = %.2f; c_minus = %.2f \n' %
        #             (c_bar,this_RR_reshape_DNS.mean(),self.c_plus,self.c_minus))
        #fig_name = join(self.output_path,'c_bar_%.4f_wrinkl_%.3f_filter_%i_%s.png' % (c_bar, wrinkling,self.filter_width, self.case))
        #plt.savefig(fig_name)

        print('c_bar: ', c_bar)
        print(' ')



    # compute self.delta_x_0
    def get_delta_0(self,c):
        '''
        :param c: usually c_0
        :param m: steepness of the flame front; not sure how computed
        :return: computes self.delta_x_0, needed for c_plus and c_minus
        '''
        return (1 - c ** self.m) / (1-c)

    def compute_c_0(self,c_bar):
        '''
        :param c: c_bar.
        :param self.Delta: this is the scaled filter width. -> is ocmputed before
        :param m: steepness of the flame front; not sure how computed
        :return: c_0
        '''

        self._c_0=(1-c_bar)*np.exp(-self.Delta/7) + (1 - np.exp(-self.Delta/7))*(1-np.exp(-2*(1-c_bar)*self.m))

    def compute_c_minus(self,c_bar):
        '''
        :param c_bar:
        :return: self.c_minus
        '''
        # update c_0 and delta_0
        self.compute_c_0(c_bar = c_bar)
        this_delta_0 = self.get_delta_0(c=(1-self._c_0))

        self.c_minus = (np.exp(c_bar * this_delta_0 * (1-c_bar) * self.Delta)-1) / \
                       (np.exp(this_delta_0 * (1-self._c_0)*self.Delta) -1)

    def compute_c_plus(self,c_bar):
        '''
        :return: self.c_plus
        '''
        # update c_minus
        self.compute_c_minus(c_bar)

        self.c_plus = (self.c_minus * np.exp(self.Delta)) / (1 + self.c_minus*(np.exp(self.Delta)-1))



# if __name__ == "__main__":
#
#     print('Starting 1bar case!')
#     filter_widths=[16,32]
#
#     for f in filter_widths:
#         # RENEW!!!
#         bar1 = data_binning_PDF(case='1bar',m=4.8,  alpha=0.81818, beta=6, bins=20)
#         bar1.dask_read_transform()
#         print('\nRunning with filter width: %i' % f)
#         bar1.run_analysis_wrinkling(filter_width=f, interval=8, histogram=True)
#         del bar1





