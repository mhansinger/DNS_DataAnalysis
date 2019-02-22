'''
This is to read in the binary data File for the high pressure bunsen data and perform post processing

@author: mhansinger

last change: 8.1.2019
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import dask.dataframe as dd
import dask.array as da
from numba import jit
#from mayavi import mlab
# to free memory
import gc
import time


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
            self.Nx = 250
            self.bfact = 7128.3 #7364.0
            self.Re = 1000
            self.delta_x = 1/611
            self.p = 10
        else:
            raise ValueError('This case does not exist!\nOnly: 1bar, 5bar, 10bar\n')

        # for reaction rates
        self.alpha = alpha
        self.beta = beta
        self.m = m

        # normalizing pressure
        self.p_0 = 1

        # TO BE COMPUTED
        self.this_c_bar = None
        self.c_tilde = None
        self.c_bar_old = 0
        self.f = None

        self.c_0 = None
        self.c_plus = None
        self.c_minus = None
        self.pc_by_dc = None
        self.analytical_hist = None

        # SCHMIDT NUMBER
        self.Sc = 0.7

        # OFFSET EPS TO AVOID SINGULARITY AT C_PLUS AND C_MINUS FOR CALCULATION OF p(c)
        self.eps = 0.000025

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
        ''' Dask reader '''

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
    def run_analysis(self,filter_width = 8, interval = 2, c_min_thresh=0.005, c_max_thresh = 1.0, histogram=True):
        # run the analysis without computation of wrinkling factor -> planar flame (dummy case)

        self.filter_width =filter_width
        # self.threshold = c_min_threshold
        self.interval = interval
        # self.c_rho_max = c_max_threshold

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta = (self.Sc * self.Re) / np.sqrt(self.p/self.p_0) * self.delta_x * self.filter_width

        # LOOP OVER THE 3 DIMENSIONS
        count = 0
        for k in range(self.filter_width-1,self.Nx,self.interval):
            for j in range(self.filter_width - 1, self.Nx, self.interval):
                for i in range(self.filter_width - 1, self.Nx, self.interval):

                    # this is the current data cube which constitutes the LES cell
                    self.this_rho_c_set = self.rho_c_data_da[i-self.filter_width:i ,j-self.filter_width:j, k-self.filter_width:k].compute()
                    # get the density for the relevant points! it is stored in a different file!
                    self.this_rho_set = self.rho_data_da[i - self.filter_width:i, j - self.filter_width:j,
                                        k - self.filter_width:k].compute()
                    self.this_c_set = self.this_rho_c_set / self.this_rho_set

                    # c_bar is computed
                    self.this_c_bar = self.this_c_set.mean()

                    # CRITERIA BASED ON C_BAR IF DATA IS FURTHER ANALYSED
                    # (CONSIDER DATA WHERE THE FLAME IS, THROW AWAY EVERYTHING ELSE)
                    if c_min_thresh < self.this_c_bar <= c_max_thresh and self.c_bar_old != self.this_c_bar:

                        self.c_bar_old = self.this_c_bar

                        self.compute_RR(i,j,k,histogram)


    @jit
    def compute_RR(self,i,j,k,histogram):
        #compute c_bar without wrinkling factor

        this_rho_c_reshape = self.this_rho_c_set.reshape(self.filter_width**3)
        this_rho_reshape = self.this_rho_set.reshape(self.filter_width**3)

        this_rho_c_mean = this_rho_c_reshape.mean()

        this_rho_mean = this_rho_reshape.mean()

        # c without density
        this_c_reshape = self.this_c_set.reshape(self.filter_width**3) #this_rho_c_reshape / this_rho_reshape

        # compute c_tilde: mean(rho*c)/mean(rho)
        self.this_c_tilde = this_rho_c_mean / this_rho_mean

        # compute the reaction rate of each cell     Eq (2.28) Senga documentation
        # note that for Le=1: c=T, so this_c_reshape = this_T_reshape
        exponent = - self.beta*(1-this_c_reshape) / (1 - self.alpha*(1 - this_c_reshape))
        this_RR_reshape_DNS = self.bfact*this_rho_reshape*(1-this_c_reshape)*np.exp(exponent)

        this_RR_reshape_DNS[this_RR_reshape_DNS<0] = 0

        # construct empty data array and fill it
        data_arr = np.zeros((self.filter_width ** 3, len(self.col_names)))
        data_arr[:, 0] = self.this_c_tilde
        data_arr[:, 1] = this_rho_mean
        data_arr[:, 2] = this_rho_c_reshape
        data_arr[:, 3] = this_rho_reshape
        data_arr[:, 4] = this_RR_reshape_DNS
        # data_arr[:, 5] = int(i)
        # data_arr[:, 6] = int(j)
        # data_arr[:, 7] = int(k)
        #
        data_df = pd.DataFrame(data_arr, columns=self.col_names)
        file_name = 'c_tilde_%.5f_filter_%i_%s.csv' % (self.this_c_tilde, self.filter_width, self.case)

        #############################################
        # computation of the integration boundaries
        self.compute_c_minus()
        self.compute_c_plus()

        print('\nc_bar: %.4f' % self.this_c_bar)
        print('c_plus: ', self.c_plus)
        print('c_minus: ', self.c_minus)
        print('c_0: ', self.c_0)
        print('delta_0: ', self.get_delta_0(self.c_0))
        print('Delta_LES: ', self.Delta)
        #############################################

        if self.write_csv:
            data_df.to_csv(join(self.output_path, file_name), index=False)

        if histogram:
            self.plot_histograms_intervals(this_rho_c_reshape=this_rho_c_reshape,
                                           this_rho_reshape=this_rho_reshape,this_RR_reshape_DNS=this_RR_reshape_DNS)

    def compute_analytical_pdf(self):
        # this is to compute the analytical PDF based on c_plus and c_minus

        # compute appropriate vector with c values between c_minus and c_plus:
        points = 200#self.filter_width**3 / 10

        self.this_c_vector = np.linspace(self.c_minus,self.c_plus,points)

        self.analytical_c_pdf = ( 1 / (self.Delta * self.this_c_vector * (1 - self.this_c_vector ** self.m + self.eps))) \
                                  * (self.Re * self.Sc * self.p_0/self.p)

        # get the gradient of c_vector (equidistant in this case)
        dc = self.this_c_vector[1]-self.this_c_vector[0]

        self.pc_by_dc = self.analytical_c_pdf * dc
        self.pc_by_dc_2 = self.analytical_c_pdf[2:-2] * dc

        # check the INT(self.analytical_c_pdf) dc = 1
        self.Integral = np.trapz(self.pc_by_dc,dx= 1/points)#self.pc_by_dc.sum() #np.trapz(self.analytical_c_pdf,dx= 1/points)
        print('Integral %.3f' % self.Integral)
        print('Integral %.3f\n' % self.pc_by_dc_2.sum())

        factor = self.bins * 100
        self.analytical_hist = np.zeros(factor)

        # DEFINE THE CHUNK SIZE
        chunk_size = int(points/factor)

        #loop over the data:
        for i,_ in enumerate(self.analytical_hist):
            if i==0:
                self.analytical_hist[i] = self.analytical_c_pdf[i+1:i+chunk_size-1].sum()
            else:
                self.analytical_hist[i] = self.analytical_c_pdf[1:i + chunk_size].sum()

        normalizer = self.analytical_hist.sum()
        self.analytical_hist = self.analytical_hist/normalizer


    def compute_analytical_RR(self):
        # computes the analytical RR (omega_bar) based on the interval boundaries c_minus and c_plus
        self.RR_analytical = ((self.c_plus** (self.m + 1) - self.c_minus** (self.m + 1))  / self.Delta) * (self.Re * self.Sc * self.p_0 / self.p)


    def plot_histograms_intervals(self,this_rho_c_reshape,this_rho_reshape,this_RR_reshape_DNS,wrinkling=1):
        # plot c, RR, and integration boundaries c_plus, c_minus

        # fig, (ax1) = plt.subplots(ncols=1, figsize=(16, 6))
        # fig, (ax1) = plt.subplots(ncols=1)
        fig, (ax0, ax1) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 4]},  figsize=(14, 5)) #plt.subplots(ncols=2)#, figsize=(10, 4))

        # compute the mean reaction rate for the single bins
        sorted_rho_c_reshape = np.sort(this_rho_c_reshape)
        idx = np.unravel_index(np.argsort(this_rho_c_reshape, axis=None), this_rho_c_reshape.shape)
        sorted_RR_reshape_DNS = np.array([x for y, x in sorted(zip(this_rho_c_reshape,this_RR_reshape_DNS))])

        hist, bin_edges = np.histogram(this_rho_c_reshape/this_rho_reshape, bins=self.bins)

        # get the analytical c pdf
        self.compute_analytical_pdf()

        # compute the analytical RR
        self.compute_analytical_RR()

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
            counter =+ val

        # GARBAGE COLLECT AND FREE MEMORY
        gc.collect()
        #print('RR_LES_mean: ', RR_LES_mean)

        ###################################

        this_c_reshape = this_rho_c_reshape/this_rho_reshape
        ax1.set_title('c histogram / pdf')
        ax1.set_xlim(0,1)

        ax2 = ax1.twinx()
        color = 'g'
        ax2.set_ylabel('Reaction Rate', color=color)

        ax1.hist(this_c_reshape, bins=self.bins, density=1, edgecolor='black', linewidth=0.7)
        ax2.scatter(this_c_reshape, this_RR_reshape_DNS, color='k', s=2.8)
        ax2.scatter(this_c_reshape, this_RR_reshape_DNS, color=color, s=2.5)
        ax2.set_ylim(0,90)

        # include analytical pdf
        ax1.plot(self.this_c_vector, self.pc_by_dc, color = 'r', linewidth=2)

        # binning of analytical pdf
        # ax1.hist(self.analytical_hist, bins=self.bins, density=1, color='red', alpha=0.5,edgecolor='black', linewidth=0.7)


        # add legend
        ax1.legend(['c pdf', 'c histogram'])

        fig.tight_layout()

        text_str = 'c_bar = %.5f\nc_plus = %.5f\nc_minus = %.5f\nfilter = %i\nDelta_LES = %.1f\nm = %.1f\nRR_mean_DNS = %.1f\nRR_mean_PDF = %.1f\nInt. p(c)dc = %.0f' \
                   % (self.this_c_bar ,self.c_plus,self.c_minus,self.filter_width,self.Delta,self.m,this_RR_reshape_DNS.mean(),self.RR_analytical,self.pc_by_dc_2.sum())

        ax0.text(0.1, 0.5, text_str, size=15)
        ax0.axis('off')

        # plt.text(text_str, fontsize=14, bbox=dict(facecolor='none', edgecolor='black'))
        # plt.subplots_adjust(left=0.1)

        fig_name = join(self.output_path,'c_bar_%.4f_wrinkl_%.3f_filter_%i_%s.png' % (self.this_c_bar, wrinkling,self.filter_width, self.case))
        plt.savefig(fig_name)
        plt.close('all')
        #plt.show(block=False)

    # compute self.delta_x_0
    def get_delta_0(self,c):
        '''
        :param c: usually c_0
        :param m: steepness of the flame front; not sure how computed
        :return: computes self.delta_x_0, needed for c_plus and c_minus
        '''
        return (1 - c ** self.m) / (1-c)


    def compute_c_0(self):
        '''
        :param c: c_bar.
        :param self.Delta: this is the scaled filter width. -> is ocmputed before
        :param m: steepness of the flame front; not sure how computed
        :return: c_0
        '''

        self.compute_f()        # update the f value

        c = self.this_c_bar
        Delta = self.Delta
        m = self.m

        self.c_0=(1 - self.this_c_bar)*self.f + \
                 (1 - self.f) * \
                 (1 - np.exp(- 2 * (1 - self.this_c_bar) * self.m))


    def compute_c_minus(self):
        '''
        :return: self.c_minus
        Eq. 37
        '''
        # update c_0 and delta_0
        self.compute_c_0()
        this_delta_0 = self.get_delta_0(c=(1-self.c_0))

        self.c_minus = (np.exp(self.this_c_bar * this_delta_0 * self.Delta)-1) / \
                       (np.exp(this_delta_0 * self.Delta) - 1)


    def compute_c_plus(self):
        '''
        :return: self.c_plus
        '''
        # update c_minus
        self.compute_c_minus()

        self.c_plus = (1 + (-1 + self.c_minus ** (-self.m)) * np.exp(-self.Delta * self.m)) ** (-1/self.m)


    def compute_f(self):
        '''
        :return: f value, APF. 28
        '''
        a = 8.25
        b = 0.75
        g = 1/3
        d = e = 4

        self.f = np.exp(- self.Delta / a) - b * self.this_c_bar ** 2 * np.exp(-g * self.m) * np.exp(-((self.Delta - d)/ e) ** 2)






