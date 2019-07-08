'''
This is to read in the binary data File for the high pressure bunsen data

@author: mhansinger

last change: July 2019
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import dask.dataframe as dd
import dask.array as da
from numba import jit
from mayavi import mlab
# to free memory
import gc
import dask
import scipy as sp
import scipy.ndimage
import dask.array as da

#TODO
# anpassen des LES Gradienten!
# implement Gauss filter?

class data_binning_PDF(object):

    def __init__(self, case, bins):
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

        if self.case is '1bar':
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
            m = 4.4545
            beta=6
            alpha=0.81818
        elif self.case=='5bar':
            self.Nx = 560
            self.bfact = 7128.3
            self.Re = 500
            self.delta_x = 1/432
            self.p = 5
            m = 4.4545
            beta=6
            alpha=0.81818
        elif self.case=='10bar':
            self.Nx = 795
            self.bfact = 7128.3
            self.Re = 1000
            self.delta_x = 1/611
            self.p = 10
            m = 4.4545
            beta=6
            alpha=0.81818
        elif self.case is 'dummy_planar_flame':
            # this is a dummy case with 50x50x50 entries!
            print('\n################\nThis is the dummy test case!\n################\n')
            self.Nx = 150
            self.bfact = 7364.0
            self.Re = 100
            self.delta_x = 1/188
            self.p = 1
        elif self.case.startswith('NX512'):
            # check: Parameter_PlanarFlame.xlsx
            self.Nx = 512
            self.bfact = 3675
            self.Re = 50
            self.delta_x = 1/220    # Klein nochmal fragen! -> 220 sollte stimmen!
            self.p = 1
            m = 4.4545
            beta=6
            alpha=0.81818
        elif self.case is 'planar_flame_test':
            # check: Parameter_PlanarFlame.xlsx
            print('\n################\nThis is the laminar planar test case!\n################\n')
            self.Nx = 512
            self.bfact = 3675
            self.Re = 50
            self.delta_x = 1/220    # Klein nochmal fragen! -> 220 sollte stimmen!
            self.p = 1
            m = 4.4545
            beta=6
            alpha=0.81818
        else:
            raise ValueError('This case does not exist!\nOnly: 1bar, 5bar, 10bar\n')

        # for reaction rates
        self.alpha = alpha
        self.beta = beta
        self.m = m

        # # will be overwritten with the correct value later !
        # self.filter_width = 0

        # normalizing pressure
        self.p_0 = 1

        # TO BE COMPUTED
        self.c_bar = None
        self.rho_bar = None

        self._c_0 = None
        self.c_plus = None
        self.c_minus = None
        self.omega_bar_model = None
        self.wrinkling_factor=None
        self.RR_DNS = None              #Reaction rate computed from DNS Data

        # Variables for FILTERING
        self.c_filtered = np.zeros((self.Nx,self.Nx,self.Nx))
        self.rho_filtered = np.zeros((self.Nx,self.Nx,self.Nx))

        # SCHMIDT NUMBER
        self.Sc = 0.7

        # DELTA_LES: NORMALIZED FILTERWIDTH
        self.Delta_LES = None # --> is computed in self.run_analysis !
        self.gauss_kernel = None

        #Data array to store the results
        self.dataArray_np = np.zeros(7)
        self.data_flag = True

        # checks if output directory exists
        self.output_path = join(case,'output_test')
        if os.path.isdir(self.output_path) is False:
            os.mkdir(self.output_path)

        self.col_names = ['c_tilde','rho_bar','c_rho','rho','reactionRate']

        print('Case: %s' % self.case)
        print('Nr. of grid points: %i' % self.Nx)
        print("Re = %f" % self.Re)
        print("Sc = %f" % self.Sc)
        print("dx_DNS = %f" % self.delta_x)

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

        print('Read in data...')
        # transform the data into an array and reshape it to 3D
        self.rho_data_np = self.data_rho.to_dask_array(lengths=True).reshape(self.Nx,self.Nx,self.Nx).compute()
        self.rho_c_data_np = self.data_rho_c.to_dask_array(lengths=True).reshape(self.Nx,self.Nx,self.Nx).compute()

        # progress variable
        self.c_data_np=  self.rho_c_data_np / self.rho_data_np

    @jit
    def run_analysis(self,filter_width = 8, interval = 2, threshold=0.005, c_rho_max = 0.1818, histogram=True):
        # run the analysis without computation of wrinkling factor -> planar flame (dummy case)

        self.filter_width =filter_width
        self.threshold = threshold
        self.interval = interval
        self.c_rho_max = c_rho_max

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)

        count = 0
        for k in range(self.filter_width-1,self.Nx,self.interval):
            for j in range(self.filter_width - 1, self.Nx, self.interval):
                for i in range(self.filter_width - 1, self.Nx, self.interval):

                    # TEST VERSION
                    # this is the current data cube which constitutes the LES cell
                    this_rho_c_set = self.rho_c_data_np[i-self.filter_width:i ,j-self.filter_width:j, k-self.filter_width:k].compute()

                    #print(this_rho_c_set)

                    # check if threshold condition is reached
                    # -> avoid computations where c_bar is either 0 or 1 as there is no flame front
                    if (this_rho_c_set > self.threshold).any() and (this_rho_c_set < self.c_rho_max).all():

                        #print('If criteria erreicht!')
                        #compute c-bar

                        self.compute_cbar(this_rho_c_set,i,j,k,histogram)

    def apply_gauss_filter(self):
        # filter c and rho data set with gauss filter function
        print('Apply Gaussian filter...')
        sigma_xy = [self.filter_width, self.filter_width ,self.filter_width]

        self.rho_filtered = sp.ndimage.filters.gaussian_filter(self.rho_data_np, sigma_xy, truncate=1.0, mode='reflect')

        self.c_filtered = sp.ndimage.filters.gaussian_filter(self.rho_c_data_np/self.rho_data_np, sigma_xy, truncate=1.0, mode='reflect')


    @jit
    def run_analysis_wrinkling(self,filter_width = 8, interval = 2, c_min_thresh=0.01, c_max_thresh = 0.99, histogram=False, write_csv=False):
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes

        self.write_csv = write_csv
        self.filter_width = int(filter_width)
        #self.threshold = threshold
        self.interval = interval

        # filter the c and rho field
        self.apply_gauss_filter()

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES= self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        # Set the Gauss kernel
        self.set_gaussian_kernel()

        # loop over the DNS Data
        count = 0
        if self.case is 'planar_flame_test':
            # for k in range(self.filter_width - 1, self.Nx, self.interval):
            #     for j in range(120,150): #range(self.filter_width - 1, self.Nx, self.interval):
            #         for i in range(120,150):
            #
            #             # c_bar is computed
            #             self.c_bar = self.c_filtered[i,j,k]
            #             self.rho_bar = self.rho_filtered[i,j,k]
            #
            #             # CRITERIA BASED ON C_BAR IF DATA IS FURTHER ANALYSED
            #             # (CONSIDER DATA WHERE THE FLAME IS, THROW AWAY EVERYTHING ELSE)
            #             if c_min_thresh < self.c_bar <= c_max_thresh:  # and self.c_bar_old != self.c_bar:
            #
            #                 self.compute_wrinkling_RR(i, j, k, histogram)
            #
            #                 if self.data_flag:
            #                     # update the data array for output
            #                     this_data_vec = np.array([self.c_bar, self.wrinkling_factor, np.mean(self.RR_DNS),
            #                                               np.mean(self.RR_DNS_Pfitz), self.omega_bar_model, self.c_plus,
            #                                               self.c_minus])
            #                     # append the data
            #                     self.dataArray_np = np.vstack([self.dataArray_np, this_data_vec])
            #
            #             # obj = delayed(self.compute_loop_analysis)(i, j, k, c_min_thresh, c_max_thresh, histogram)
            #             # obj.compute()

            # compute the wrinkling factor
            self.get_wrinkling()
            self.compute_Pfitzner_model()


        else:
            for k in range(self.filter_width - 1, self.Nx, self.interval):
                for j in range(self.filter_width - 1, self.Nx, self.interval):
                    for i in range(self.filter_width - 1, self.Nx, self.interval):

                        # # this is the current data cube which constitutes the LES cell
                        # self.this_rho_c_set = self.rho_c_data_np[i - self.filter_width:i, j - self.filter_width:j,
                        #                       k - self.filter_width:k]
                        # # get the density for the relevant points! it is stored in a different file!
                        # self.this_rho_set = self.rho_data_np[i - self.filter_width:i, j - self.filter_width:j,
                        #                     k - self.filter_width:k]
                        # self.this_c_set = self.this_rho_c_set / self.this_rho_set
                        #
                        # # c_bar is computed
                        # self.c_bar = self.this_c_set.mean()
                        # self.rho_bar = self.this_rho_set.mean()
                        #
                        # # CRITERIA BASED ON C_BAR IF DATA IS FURTHER ANALYSED
                        # # (CONSIDER DATA WHERE THE FLAME IS, THROW AWAY EVERYTHING ELSE)
                        # if c_min_thresh < self.c_bar <= c_max_thresh:  # and self.c_bar_old != self.c_bar:
                        #
                        #     self.compute_wrinkling_RR(i, j, k, histogram)
                        #
                        #     if self.data_flag:
                        #         # update the data array for output
                        #         this_data_vec = np.array([self.c_bar, self.wrinkling_factor, np.mean(self.RR_DNS),
                        #                                   np.mean(self.RR_DNS_Pfitz), self.omega_bar_model, self.c_plus,
                        #                                   self.c_minus])
                        #         # append the data
                        #         self.dataArray_np = np.vstack([self.dataArray_np, this_data_vec])

                        obj = delayed(self.compute_loop_analysis)(i, j, k, c_min_thresh, c_max_thresh, histogram)
                        obj.compute()

        dataArray_da = da.vstack([self.c_filtered.reshape(self.Nx**3),
                                   self.wrinkling_factor.reshape(self.Nx**3),
                                   self.omega_model_cbar.reshape(self.Nx**3),
                                   self.omega_DNS_filtered.reshape(self.Nx**3),
                                   self.c_plus.reshape(self.Nx**3),
                                   self.c_minus.reshape(self.Nx**3)])

        # write data to csv file
        filename = join(self.case,'filter_width_'+str(self.filter_width)+'.csv')
        dataArray_dd = dd.DataFrame(dataArray_da,
                                    ['c_bar','wrinkling','omega_model','omega_DNS_filtered','c_plus','c_minus'])

        # filter the data set and remove unecessary entries
        dataArray_dd = dataArray_dd[dataArray_dd['c_bar'] > 0.001]
        dataArray_dd = dataArray_dd[dataArray_dd['c_bar'] < 0.99]

        # print('Writing output to csv ...')
        # dataArray_dd.to_csv(filename,index=False)
        # print('Data has been written.')


    def compute_loop_analysis(self, i, j, k, c_min_thresh, c_max_thresh, histogram):
        # this is the current data cube which constitutes the LES cell
        self.this_rho_c_set = self.rho_c_data_np[i - self.filter_width:i, j - self.filter_width:j,
                              k - self.filter_width:k]
        # get the density for the relevant points! it is stored in a different file!
        self.this_rho_set = self.rho_data_np[i - self.filter_width:i, j - self.filter_width:j,
                            k - self.filter_width:k]
        self.this_c_set = self.this_rho_c_set / self.this_rho_set

        # c_bar is computed
        self.c_bar = self.this_c_set.mean()
        self.rho_bar = self.this_rho_set.mean()

        # CRITERIA BASED ON C_BAR IF DATA IS FURTHER ANALYSED
        # (CONSIDER DATA WHERE THE FLAME IS, THROW AWAY EVERYTHING ELSE)
        if c_min_thresh < self.c_bar <= c_max_thresh:  # and self.c_bar_old != self.c_bar:

            self.compute_wrinkling_RR(i, j, k, histogram)

            if self.data_flag:
                # update the data array for output
                this_data_vec = np.array(
                    [self.c_bar, self.wrinkling_factor, self.RR_DNS, self.RR_DNS_Pfitz, self.omega_bar_model,
                     self.c_plus, self.c_minus])
                self.dataArray_np = np.vstack([self.dataArray_np, this_data_vec])


    def plot_histograms(self,c_tilde,this_rho_c_reshape,this_rho_reshape,this_RR_reshape_DNS):
        # plot the c_tilde and c, both normalized

        c_max = 0.182363

        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))
        ax0.hist(this_rho_c_reshape,bins=self.bins,normed=True,edgecolor='black', linewidth=0.7)
        ax0.set_title('rho*c')
        ax0.set_xlim(0,c_max)

        # compute the mean reaction rate for the single bins
        sorted_rho_c_reshape = np.sort(this_rho_c_reshape)
        idx = np.unravel_index(np.argsort(this_rho_c_reshape, axis=None), this_rho_c_reshape.shape)
        sorted_RR_reshape_DNS = np.array([x for y, x in sorted(zip(this_rho_c_reshape,this_RR_reshape_DNS))])

        hist, bin_edges = np.histogram(this_rho_c_reshape, bins=self.bins)

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

        print('RR_LES_mean: ', RR_LES_mean)

        ###################################

        this_c_reshape = this_rho_c_reshape/this_rho_reshape
        #ax1.hist(this_c_reshape,bins=self.bins,normed=True,edgecolor='black', linewidth=0.7)
        ax1.set_title('c')
        ax1.set_xlim(0,1)

        ax2 = ax1.twinx()
        color = 'r'
        ax2.set_ylabel('Reaction Rate', color=color)
        # ax2.scatter(this_c_reshape, this_RR_reshape_DNS,color=color,s=0.9)
        # ax1.hist(this_RR_reshape_DNS/this_RR_reshape_DNS.max(), bins=self.bins,normed=True,color=color,alpha=0.5,edgecolor='black', linewidth=0.7)
        ax1.hist(this_c_reshape, bins=self.bins, normed=True, edgecolor='black', linewidth=0.7)
        ax2.scatter(this_c_reshape, this_RR_reshape_DNS, color=color, s=0.9)
        ax2.set_ylim(0,90)

        #fig.tight_layout()
        plt.suptitle('c_tilde = %.3f; c_bar = %.3f; wrinkling = %.3f; RR_mean_DNS = %.2f; RR_mean_LES = %.2f\n' %
                     (c_tilde,self.c_bar,self.wrinkling_factor,this_RR_reshape_DNS.mean(),RR_LES_mean) )
        fig_name = join(self.output_path,'c_bar_%.4f_wrinkl_%.3f_filter_%i_%s.png' % (self.c_bar, self.wrinkling_factor,self.filter_width, self.case))
        plt.savefig(fig_name)

        print('c_bar: ', self.c_bar)
        print(' ')

        # plt.figure()
        # plt.title('c_tilde = %.5f' % c_tilde)
        # plt.subplot(211)
        # plt.hist(this_rho_c_reshape,bins=self.bins,normed=True)
        # plt.title('rho x c')
        #
        # plt.subplot(221)
        # plt.hist(this_rho_c_reshape/this_rho_reshape,bins=self.bins,normed=True)
        # plt.title('c')


    def plot_histograms_intervals(self,c_tilde,this_rho_c_reshape,this_rho_reshape,c_bar,this_RR_reshape_DNS,wrinkling=1):
        # plot c, RR, and integration boundaries c_plus, c_minus

        fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 4))
        # ax0.hist(this_rho_c_reshape,bins=self.bins,normed=True,edgecolor='black', linewidth=0.7)
        # ax0.set_title('rho*c')
        # ax0.set_xlim(0,c_max)

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
        ax1.set_title('c')
        ax1.set_xlim(0,1)

        ax2 = ax1.twinx()
        color = 'r'
        ax2.set_ylabel('Reaction Rate', color=color)
        ax1.hist(this_c_reshape, bins=self.bins, normed=True, edgecolor='black', linewidth=0.7)
        ax2.scatter(this_c_reshape, this_RR_reshape_DNS, color=color, s=0.9)
        ax2.set_ylim(0,90)

        #fig.tight_layout()
        plt.suptitle('c_bar = %.3f;  RR_mean_DNS = %.2f; c_plus = %.2f; c_minus = %.2f \n' %
                     (c_bar,this_RR_reshape_DNS.mean(),self.c_plus,self.c_minus))
        fig_name = join(self.output_path,'c_bar_%.4f_wrinkl_%.3f_filter_%i_%s.png' % (c_bar, wrinkling,self.filter_width, self.case))
        plt.savefig(fig_name)

        print('c_bar: ', c_bar)
        print(' ')


    def set_gaussian_kernel(self):
        # wird vermutlich nicht gebraucht...
        size = int(self.filter_width)
        vector = np.linspace(-self.filter_width,self.filter_width,2*self.filter_width+1)
        x,y,z = np.meshgrid(vector, vector, vector)
        x = x * self.delta_x
        y = y * self.delta_x
        z = z * self.delta_x

        self.gauss_kernel = np.sqrt(12)/self.Delta_LES/np.sqrt(2*np.pi) * \
                            np.exp(-6*(x**2/self.Delta_LES**2 +y**2/self.Delta_LES**2 + z**2/self.Delta_LES**2))


    @jit
    def compute_wrinkling_RR(self,i,j,k,histogram):
        # this is to compute with the wrinkling factor and c_bar

        #TODO
        # this has to be re written for usage with the Gaussian filter!

        #this_rho_c_reshape = self.this_rho_c_set.reshape(self.filter_width**3)
        this_rho_reshape = self.this_rho_set.reshape(self.filter_width**3)

        #this_rho_c_mean = this_rho_c_reshape.mean()

        this_rho_mean = this_rho_reshape.mean()

        # c without density
        this_c_reshape = self.this_c_set.reshape(self.filter_width**3) #this_rho_c_reshape / this_rho_reshape

        # compute c_tilde: mean(rho*c)/mean(rho)
        #self.this_c_tilde = this_rho_c_mean / this_rho_mean

        this_rho_mean = this_rho_reshape.mean()

        # compute c_tilde: mean(rho*c)/mean(rho)
        #c_tilde = this_rho_c_mean / this_rho_mean

        # compute the reaction rate of each cell     Eq (2.28) Senga documentation
        # note that for Le=1: c=T, so this_c_reshape = this_T_reshape
        exponent = - self.beta*(1-this_c_reshape) / (1 - self.alpha*(1 - this_c_reshape))
        this_RR_reshape_DNS = self.bfact*this_rho_reshape*(1-this_c_reshape)*np.exp(exponent)

        this_RR_reshape_DNS_Pfitz  = 18.97 * ((1 - self.alpha * (1 - this_c_reshape))) ** (-1) * (1 - this_c_reshape) * np.exp(exponent)

        self.RR_DNS_Pfitz = this_RR_reshape_DNS_Pfitz
        self.RR_DNS = this_RR_reshape_DNS

        # another criteria
        if 0.01 < self.c_bar < 0.99: #0.7 < this_c_bar < 0.9:
            #print(this_c_reshape)
            # print("c_bar: ",self.c_bar)

            # COMPUTE WRINKLING FACTOR
            self.wrinkling_factor = self.get_wrinkling(i,j,k)

            # consistency check, wrinkling factor needs to be >1!
            if self.wrinkling_factor >= 1:
                self.data_flag=True

                ###############################################
                # Pfitzner's model
                self.compute_Pfitzner_model()
                # so far no values are written to files
                ###############################################

                # if self.write_csv:
                #     data_df.to_csv(join(self.output_path, file_name), index=False)

                if histogram:
                    self.plot_histograms(c_tilde=c_tilde, this_rho_c_reshape=this_rho_c_reshape,
                                         this_rho_reshape=this_rho_reshape,
                                         this_RR_reshape_DNS=this_RR_reshape_DNS)
                    # self.plot_histograms()
                # print('c_tilde: ', c_tilde)

                # plot the surface in the box if wrinkling > 10
                if self.wrinkling_factor > 100:
                    file_name_3D_plot = 'c_bar_%.4f_wrinkl_%.3f_filter_%i_%s_ISO_surface.png' % (
                    self.c_bar, self.wrinkling_factor, self.filter_width, self.case)

                    c_3D = this_c_reshape.reshape(self.filter_width,self.filter_width,self.filter_width)

                    mlab.contour3d(c_3D)
                    mlab.savefig(join(self.output_path,file_name_3D_plot))
                    mlab.close()

            else:
                print("##################")
                print("Computed Nonsense!")
                print("##################\n")
                self.data_flag=False


    def get_wrinkling(self):
        # computes the wriknling factor from resolved and filtered flame surface
        #print(i)

        grad_DNS_filtered = self.compute_filter_DNS_grad()
        grad_LES = self.compute_LES_grad()

        #compute the wrinkling factor
        print('Computing wrinkling factor ...')
        self.wrinkling_factor = grad_DNS_filtered / grad_LES

    #@dask.delayed
    def compute_DNS_grad(self):
        # computes the flame surface area in the DNS based on gradients of c of neighbour cells
        width = 1

        print('Computing DNS gradients...')

        # create empty array
        grad_c_DNS = np.zeros([self.Nx,self.Nx,self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(1,self.Nx-1):
            for m in range(1,self.Nx-1):
                for n in range(1,self.Nx-1):
                    this_DNS_gradX = (self.c_data_np[l+1, m, n] - self.c_data_np[l-1,m, n])/(2 * self.delta_x)
                    this_DNS_gradY = (self.c_data_np[l, m+1, n] - self.c_data_np[l, m-1, n]) / (2 * self.delta_x)
                    this_DNS_gradZ = (self.c_data_np[l, m, n+1]- self.c_data_np[l, m, n-1]) / (2 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_DNS_magGrad_c = np.sqrt(this_DNS_gradX**2 + this_DNS_gradY**2 + this_DNS_gradZ**2)

                    grad_c_DNS[l,m,n] = this_DNS_magGrad_c

        return grad_c_DNS

    #@dask.delayed
    def compute_LES_grad(self):
        # computes the flame surface area in the DNS based on gradients of c of neighbour cells

        print('Computing LES gradients on DNS mesh ...')

        # create empty array
        grad_c_LES = np.zeros([self.Nx, self.Nx, self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(1, self.Nx - 1):
            for m in range(1, self.Nx - 1):
                for n in range(1, self.Nx - 1):
                    this_LES_gradX = (self.c_filtered[l + 1, m, n] - self.c_filtered[l - 1, m, n]) / (2 * self.delta_x)
                    this_LES_gradY = (self.c_filtered[l, m + 1, n] - self.c_filtered[l, m - 1, n]) / (2 * self.delta_x)
                    this_LES_gradZ = (self.c_filtered[l, m, n + 1] - self.c_filtered[l, m, n - 1]) / (2 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)

                    grad_c_LES[l, m, n] = this_LES_magGrad_c

        return grad_c_LES

    def compute_LES_grad_2(self):
        # computes the flame surface area in the DNS based on gradients of c of neighbour cells

        print('Computing LES gradients on LES mesh ...')

        # create empty array
        grad_c_LES = np.zeros([self.Nx, self.Nx, self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(1, self.Nx - 1):
            for m in range(1, self.Nx - 1):
                for n in range(1, self.Nx - 1):
                    this_LES_gradX = (self.c_filtered[l + self.filter_width, m, n] - self.c_filtered[l - self.filter_width, m, n]) / (2 * self.Delta_LES)
                    this_LES_gradY = (self.c_filtered[l, m + self.filter_width, n] - self.c_filtered[l, m - self.filter_width, n]) / (2 * self.Delta_LES)
                    this_LES_gradZ = (self.c_filtered[l, m, n + self.filter_width] - self.c_filtered[l, m, n - self.filter_width]) / (2 * self.Delta_LES)
                    # compute the magnitude of the gradient
                    this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)

                    grad_c_LES[l, m, n] = this_LES_magGrad_c

        return grad_c_LES

    #@dask.delayed
    def compute_filter_DNS_grad(self):
    # compute filtered DNS reaction rate
        # create empty array
        grad_DNS_filtered = np.zeros([self.Nx, self.Nx, self.Nx])

        # compute dask delayed object
        grad_c_DNS = self.compute_DNS_grad()

        sigma_xy = [self.filter_width, self.filter_width ,self.filter_width]
        grad_DNS_filtered = sp.ndimage.filters.gaussian_filter(grad_c_DNS, sigma_xy, truncate=1.0, mode='reflect')

        return grad_DNS_filtered

    #@dask.delayed
    def compute_RR_DNS(self):
        # according to Pfitzner implementation
        exponent = - self.beta*(1-self.c_data_np.reshape(self.Nx**3)) / (1 - self.alpha*(1 - self.c_data_np.reshape(self.Nx**3)))
        #this_RR_reshape_DNS = self.bfact*self.rho_data_np.reshape(self.Nx**3)*(1-self.c_data_np.reshape(self.Nx**3))*np.exp(exponent)

        this_RR_reshape_DNS_Pfitz  = 18.97 * ((1 - self.alpha * (1 - self.c_data_np.reshape(self.Nx**3)))) ** (-1) \
                                     * (1 - self.c_data_np.reshape(self.Nx**3)) * np.exp(exponent)

        RR_DNS = this_RR_reshape_DNS_Pfitz.reshape(self.Nx,self.Nx,self.Nx)

        return RR_DNS

    #@dask.delayed
    def filter_RR_DNS(self):
    # compute filtered DNS reaction rate
        # create empty array

        RR_DNS = self.compute_RR_DNS()

        sigma_xy = [self.filter_width, self.filter_width ,self.filter_width]
        RR_DNS_filtered = sp.ndimage.filters.gaussian_filter(RR_DNS, sigma_xy, truncate=1.0, mode='reflect')

        return RR_DNS_filtered


    #@dask.delayed
    def compute_RR_LES(self):
        # according to Pfitzner implementation
        exponent = - self.beta*(1-self.c_filtered.reshape(self.Nx**3)) / (1 - self.alpha*(1 - self.c_filtered.reshape(self.Nx**3)))
        #this_RR_reshape_DNS = self.bfact*self.rho_data_np.reshape(self.Nx**3)*(1-self.c_data_np.reshape(self.Nx**3))*np.exp(exponent)

        this_RR_reshape_LES_Pfitz  = 18.97 * ((1 - self.alpha * (1 - self.c_data_np.reshape(self.Nx**3)))) ** (-1) \
                                     * (1 - self.c_data_np.reshape(self.Nx**3)) * np.exp(exponent)

        RR_LES = this_RR_reshape_LES_Pfitz.reshape(self.Nx,self.Nx,self.Nx)

        return RR_LES



    # added Nov. 2018: Implementation of Pfitzner's analytical boundaries
    # getter and setter for c_Mean as a protected

    def compute_flamethickness(self):
        '''
        Eq. 17 according to analytical model (not numerical!)
        :param m:
        :return: flame thicknes dth
        '''

        return (self.m + 1) ** (1 / self.m + 1) / self.m

    # compute self.delta_x_0
    def compute_delta_0(self,c):
        '''
        Eq. 38
        :param c: usually c_0
        :param m: steepness of the flame front; not sure how computed
        :return: computes self.delta_x_0, needed for c_plus and c_minus
        '''
        return (1 - c ** self.m) / (1-c)

    def compute_c_m(self,xi):
        #Eq. 12
        return (1 + np.exp(- self.m * xi)) ** (-1 / self.m)

    def compute_xi_m(self,c):
        # Eq. 13
        return 1 / self.m * np.log(c ** self.m / (1 - c ** self.m))

    def compute_s(self,c):
        '''
        EVTL MUSS FILTERWIDTH NOCH SKALIERT WERDEN!
        Sc*Re/sqrt(p/p0)
        Eq. 39
        :param c:
        :param Delta_LES:
        :return:
        '''
        s = np.exp( - self.Delta_LES / 7) * ((np.exp(self.Delta_LES / 7) - 1) * np.exp(2 * (c - 1) * self.m) + c)
        return s

    # compute the values for c_minus
    def compute_c_minus(self):
        # Eq. 40
        this_s = self.compute_s(self.c_filtered.reshape(self.Nx**3))
        this_delta_0 = self.compute_delta_0(this_s)

        self.c_minus = (np.exp(self.c_filtered.reshape(self.Nx**3)* this_delta_0 * self.Delta_LES) - 1) / \
                       (np.exp(this_delta_0 * self.Delta_LES) - 1)


    def compute_c_plus(self):
        '''
        :param c: c_minus
        :return:
        Eq. 13
        '''
        this_xi_m = self.compute_xi_m(self.c_minus)

        xi_plus_Delta = this_xi_m + self.Delta_LES
        self.c_plus = self.compute_c_m(xi_plus_Delta)

    def compute_model_omega_bar(self):
        '''
        :param c_plus:
        :param c_minus:
        :param Delta:
        :return: omega Eq. 29
        '''

        return (self.c_plus ** (self.m + 1) - self.c_minus ** (self.m + 1)) / self.Delta_LES


    def model_omega(self,c):
        '''
        Eq. 14
        :param c:
        :return: Computes the omega from the model for c_bar!
        '''

        return (self.m + 1) * (1 - c ** self.m) * c ** (self.m + 1)


    def analytical_omega(self, c):
        '''
        Eq. 4
        :param alpha:
        :param beta:
        :param c:
        :return: computes the analytical omega for given c_bar!
        '''
        exponent = - (self.beta * (1 - c)) / (1 - self.alpha * (1 - c))
        Eigenval = 18.97  # beta**2 / 2 + beta*(3*alpha - 1.344)
        # --> Eigenval ist wahrscheinlich falsch!

        #print('Lambda:', Eigenval)

        # om_Klein = self.bfact*self.rho_bar*(1-c)*np.exp(exponent)
        om_Pfitzner = Eigenval * ((1 - self.alpha * (1 - c))) ** (-1) * (1 - c) * np.exp(exponent)

        return om_Pfitzner


    def compute_Pfitzner_model(self):
        '''
        computes the model values in sequential manner
        '''

        self.compute_c_minus()
        #print('c_minus is: ',self.c_minus)

        self.compute_c_plus()
        #print('c_plus is: ', self.c_plus)

        self.omega_model_cbar = self.compute_model_omega_bar()

        #self.omega_model_cbar = self.model_omega(self.c_filtered.reshape(self.Nx**3))
        omega_DNS_unfiltered = self.analytical_omega(self.c_data_np.reshape(self.Nx**3)).reshape(self.Nx,self.Nx,self.Nx)

        # filter the DNS reaction rate
        self.omega_DNS_filtered = sp.ndimage.filters.gaussian_filter(omega_DNS_unfiltered, [self.Nx, self.Nx, self.Nx], truncate=1.0, mode='reflect')







