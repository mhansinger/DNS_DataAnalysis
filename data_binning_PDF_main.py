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
#from numba import jit
#from mayavi import mlab
# to free memory
import gc
import dask
import scipy as sp
import scipy.ndimage
import dask.array as da
import sys
from scipy import special, interpolate
from skimage import measure
from joblib import delayed, Parallel
import time
from progress.bar import ChargingBar
import itertools
import copy
# for numerical integration
from scipy.integrate import simps


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

        #self.data_c = None
        self.data_rho = None
        self.case = case
        self.bins = bins

        # Filter width of the LES cell: is filled later
        self.filter_width = None

        self.every_nth = None

        # gradient of c on the DNS mesh
        self.grad_c_DNS = None

        # gradient of c on the LES mesh
        self.grad_c_LES = None

        if self.case is '1bar':
            # NUMBER OF DNS CELLS IN X,Y,Z DIRECTION
            self.Nx = 250
            # PARAMETER FOR REACTION RATE
            self.bfact = 7364.0
            # REYNOLDS NUMBER
            self.Re = 399 #100
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
            self.Re = 892 # 500
            self.delta_x = 1/432
            self.p = 5
            m = 4.4545
            beta=6
            alpha=0.81818
        elif self.case=='10bar':
            self.Nx = 795
            self.bfact = 7128.3
            self.Re = 1262 #1000
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

        # Variables for FILTERING
        self.c_filtered = np.zeros((self.Nx,self.Nx,self.Nx))
        self.rho_filtered = np.zeros((self.Nx,self.Nx,self.Nx))
        self.c_filtered_clipped = np.zeros((self.Nx,self.Nx,self.Nx))       # c für wrinkling factor nur zw 0.75 und 0.85

        # SCHMIDT NUMBER
        self.Sc = 0.7

        # DELTA_LES: NORMALIZED FILTERWIDTH
        self.Delta_LES = None # --> is computed in self.run_analysis !
        self.gauss_kernel = None
        self.filter_type = None

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
        self.c_data_np = self.rho_c_data_np / self.rho_data_np
        self.c_data_reduced_np = self.rho_c_data_np / self.rho_data_np      # reduce c between 0.75 and 0.85

    def apply_filter(self,data):
        # filter c and rho data set with gauss filter function
        #print('Apply Gaussian filter...')

        # check if data is 3D array
        try:
            assert type(data) == np.ndarray
        except AssertionError:
            print('Only np.ndarrays are allowed in Gauss_filter!')

        if len(data.shape) == 1:
            data = data.reshape(self.Nx,self.Nx,self.Nx)


        if self.filter_type == 'GAUSS':
            self.sigma_xyz = [int(self.filter_width/2), int(self.filter_width/2) ,int(self.filter_width/2)]
            data_filtered = sp.ndimage.filters.gaussian_filter(data, self.sigma_xyz, truncate=1.0, mode='reflect')
            return data_filtered

        elif self.filter_type == 'TOPHAT':
            data_filtered = sp.ndimage.filters.uniform_filter(data, [self.filter_width,self.filter_width,self.filter_width],mode='reflect')
            return data_filtered

        else:
            sys.exit('No fitler type provided ...')


    #@jit(nopython=True, parallel=True)
    def run_analysis_wrinkling(self,filter_width ,filter_type, c_analytical=False, Parallel=False, every_nth=1):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :param Parallel: use False
        :param every_nth: every nth DNS point to compute the isoArea
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        # joblib parallel computing of c_iso
        self.Parallel = Parallel

        self.every_nth = int(every_nth)

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        # filter the c and rho field
        print('Filtering c field ...')
        self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

        # # reduce c for computation of conditioned wrinkling factor
        # self.reduce_c(c_min=0.75,c_max=0.85)
        # self.c_filtered_reduced = self.apply_filter(self.c_data_reduced_np)

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)

        # Set the Gauss kernel
        self.set_gaussian_kernel()

        # compute the wrinkling factor
        self.get_wrinkling()
        self.compute_Pfitzner_model()

        #c_bins = self.compute_c_binning(c_low=0.8,c_high=0.9)

        start = time.time()
        if self.Parallel is True:
            isoArea_coefficient = self.compute_isoArea_parallel(c_iso=0.85)
        else:
            isoArea_coefficient = self.compute_isoArea(c_iso=0.85)
            #isoArea_coefficient = self.compute_isoArea_dynamic()

        end=time.time()
        print('computation of c_iso took %i sec: ' % int(end - start))

        # write the filtered data of the whole DNS cube only if every data point is filtered. No sparse data...(every_nth > 1)
        if self.every_nth == 1:
            # write the filtered omega and omega_model * isoArea to file
            print('writing omega DNS filtered and omega_model x isoArea to file ...')
            filename = join(self.case, 'filtered_data','omega_filtered_modeled_' + str(self.filter_width) +'_nth'+ str(self.every_nth) + '.csv')

            om_iso = self.omega_model_cbar*isoArea_coefficient
            om_wrinkl = self.omega_model_cbar*self.wrinkling_factor

            pd.DataFrame(data=np.hstack([self.omega_DNS.reshape(self.Nx**3,1),
                               self.omega_DNS_filtered.reshape(self.Nx**3,1),
                               om_iso.reshape(self.Nx**3,1),
                               om_wrinkl.reshape(self.Nx**3,1),
                               self.c_filtered.reshape(self.Nx ** 3, 1)]),
                               columns=['omega_DNS',
                                        'omega_filtered',
                                        'omega_model_by_isoArea',
                                        'omega_model_by_wrinkling',
                                        'c_bar']).to_csv(filename)

        # creat dask array and reshape all data
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx**3,1),
                                   self.wrinkling_factor.reshape(self.Nx**3,1),
                                   isoArea_coefficient.reshape(self.Nx**3,1),
                                   # self.wrinkling_factor_LES.reshape(self.Nx ** 3, 1),
                                   # self.wrinkling_factor_reduced.reshape(self.Nx ** 3, 1),
                                   # self.wrinkling_factor_LES_reduced.reshape(self.Nx ** 3, 1),
                                   self.omega_model_cbar.reshape(self.Nx**3,1),
                                   self.omega_DNS_filtered.reshape(self.Nx**3,1),
                                   #self.omega_LES_noModel.reshape(self.Nx**3,1),
                                   self.c_plus.reshape(self.Nx**3,1),
                                   self.c_minus.reshape(self.Nx**3,1)])


        if self.c_analytical is True:
            # write data to csv file
            filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_analytical.csv')
        else:
            # write data to csv file
            filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '.csv')

        self.dataArray_dd = dd.io.from_dask_array(dataArray_da,
                                             columns=['c_bar',
                                                      'wrinkling',
                                                      'isoArea',
                                                      # 'wrinkling_LES',
                                                      # 'wrinkling_reduced',
                                                      # 'wrinkling_LES_reduced',
                                                      'omega_model',
                                                      'omega_DNS_filtered',
                                                      #'omega_cbar',
                                                      'c_plus',
                                                      'c_minus'])

        # filter the data set and remove unecessary entries
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] > 0.01]
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] < 0.99]

        if self.case is 'planar_flame_test':
            self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['wrinkling'] < 1.1]

        # remove all isoArea < 1e-3 from the stored data set -> less memory
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['isoArea'] > 1e-3]
        print('isoArea < 1 is included!')

        # this is to reduce the storage size
        #self.dataArray_dd = self.dataArray_dd.sample(frac=0.3)

        print('Computing data array ...')
        self.dataArray_df = self.dataArray_dd.compute()

        print('Writing output to csv ...')
        self.dataArray_df.to_csv(filename,index=False)
        print('Data has been written.\n\n')


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


    def get_wrinkling(self,order='2nd'):
        # computes the wriknling factor from resolved and filtered flame surface
        #print(i)

        if order == '2nd':
            grad_DNS_filtered = self.compute_filter_DNS_grad()
            grad_LES = self.compute_LES_grad()
        elif order == '4th':
            grad_DNS_filtered = self.compute_filter_DNS_grad_4thO()
            grad_LES = self.compute_LES_grad_4thO()
        else:
            print('Order not defined. Only 2nd and 4th possible...')
            raise NotImplementedError

        #compute the wrinkling factor
        print('Computing wrinkling factor ...')
        self.wrinkling_factor = grad_DNS_filtered / grad_LES



    #@jit(nopython=True) #, parallel=True)
    def compute_DNS_grad(self):
        # computes the flame surface area in the DNS based on gradients of c of neighbour cells
        # magnitude of the gradient: abs(grad(c))

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

    def compute_DNS_grad_4thO(self):
        '''4th Order gradients of c from DNS data'''

        print('Computing DNS gradients...')

        # create empty array
        grad_c_DNS = np.zeros([self.Nx,self.Nx,self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(2,self.Nx-2):
            for m in range(2,self.Nx-2):
                for n in range(2,self.Nx-2):
                    this_DNS_gradX = (-self.c_data_np[l+2, m, n] + 8*self.c_data_np[l+1,m, n] - 8*self.c_data_np[l-1,m, n] + self.c_data_np[l-2, m, n])/(12 * self.delta_x)
                    this_DNS_gradY = (-self.c_data_np[l, m+2, n] + 8*self.c_data_np[l,m+1, n] - 8*self.c_data_np[l,m-1, n] + self.c_data_np[l, m-2, n])/(12 * self.delta_x)
                    this_DNS_gradZ = (-self.c_data_np[l, m, n+2] + 8*self.c_data_np[l,m, n+1] - 8*self.c_data_np[l,m, n-1] + self.c_data_np[l, m, n+2])/(12 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_DNS_magGrad_c = np.sqrt(this_DNS_gradX**2 + this_DNS_gradY**2 + this_DNS_gradZ**2)

                    grad_c_DNS[l,m,n] = this_DNS_magGrad_c

        return grad_c_DNS

    #@jit(nopython=True, parallel=True)
    def compute_DNS_grad_reduced(self):
        # computes the flame surface area in the DNS based on gradients of c of neighbour cells
        # for the reduced c
        #width = 1

        print('Computing DNS gradients for c reduced...')

        # create empty array
        grad_c_DNS = np.zeros([self.Nx,self.Nx,self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(1,self.Nx-1):
            for m in range(1,self.Nx-1):
                for n in range(1,self.Nx-1):
                    this_DNS_gradX = (self.c_data_reduced_np[l+1, m, n] - self.c_data_reduced_np[l-1,m, n])/(2 * self.delta_x)
                    this_DNS_gradY = (self.c_data_reduced_np[l, m+1, n] - self.c_data_reduced_np[l, m-1, n]) / (2 * self.delta_x)
                    this_DNS_gradZ = (self.c_data_reduced_np[l, m, n+1] - self.c_data_reduced_np[l, m, n-1]) / (2 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_DNS_magGrad_c = np.sqrt(this_DNS_gradX**2 + this_DNS_gradY**2 + this_DNS_gradZ**2)

                    grad_c_DNS[l,m,n] = this_DNS_magGrad_c

        return grad_c_DNS

    #@jit(nopython=True)
    def compute_LES_grad(self):
        # computes the flame surface area in the DNS based on gradients of c of neighbour DNS cells

        print('Computing LES gradients on DNS mesh ...')

        # create empty array
        self.grad_c_LES = np.zeros([self.Nx, self.Nx, self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(1, self.Nx - 1):
            for m in range(1, self.Nx - 1):
                for n in range(1, self.Nx - 1):
                    this_LES_gradX = (self.c_filtered[l + 1, m, n] - self.c_filtered[l - 1, m, n]) / (2 * self.delta_x)
                    this_LES_gradY = (self.c_filtered[l, m + 1, n] - self.c_filtered[l, m - 1, n]) / (2 * self.delta_x)
                    this_LES_gradZ = (self.c_filtered[l, m, n + 1] - self.c_filtered[l, m, n - 1]) / (2 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)

                    self.grad_c_LES[l, m, n] = this_LES_magGrad_c

        return self.grad_c_LES

    def compute_LES_grad_4thO(self):
        # computes the flame surface area in the DNS based on gradients of c of neighbour DNS cells
        # 4th Order version

        print('Computing LES gradients on DNS mesh ...')

        # create empty array
        self.grad_c_LES = np.zeros([self.Nx, self.Nx, self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(2, self.Nx - 2):
            for m in range(2, self.Nx - 2):
                for n in range(2, self.Nx - 2):
                    this_LES_gradX = (-self.c_filtered[l + 2, m, n] + 8*self.c_filtered[l + 1, m, n] - 8*self.c_filtered[l - 1, m, n] + self.c_filtered[l - 2, m, n]) / (12 * self.delta_x)
                    this_LES_gradY = (-self.c_filtered[l, m + 2, n] + 8*self.c_filtered[l, m+1, n] - 8*self.c_filtered[l, m-1, n] + self.c_filtered[l, m-2, n]) / (12 * self.delta_x)
                    this_LES_gradZ = (-self.c_filtered[l, m, n+2] + 8*self.c_filtered[l, m, n+1] - 8*self.c_filtered[l, m, n-1] + self.c_filtered[l, m, n-2]) / (12 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)

                    self.grad_c_LES[l, m, n] = this_LES_magGrad_c

        return self.grad_c_LES

    #@jit(nopython=True)
    def compute_LES_grad_reduced(self):
        # computes the flame surface area in the DNS based on gradients of c of neighbour DNS cells

        print('Computing LES gradients on DNS mesh for c reduced ...')

        # create empty array
        self.grad_c_LES = np.zeros([self.Nx, self.Nx, self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(1, self.Nx - 1):
            for m in range(1, self.Nx - 1):
                for n in range(1, self.Nx - 1):
                    this_LES_gradX = (self.c_filtered_reduced[l + 1, m, n] - self.c_filtered_reduced[l - 1, m, n]) / (2 * self.delta_x)
                    this_LES_gradY = (self.c_filtered_reduced[l, m + 1, n] - self.c_filtered_reduced[l, m - 1, n]) / (2 * self.delta_x)
                    this_LES_gradZ = (self.c_filtered_reduced[l, m, n + 1] - self.c_filtered_reduced[l, m, n - 1]) / (2 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)

                    self.grad_c_LES[l, m, n] = this_LES_magGrad_c

        return self.grad_c_LES

    #@jit(nopython=True)
    def compute_LES_grad_onLES(self):
        # computes the flame surface area in the DNS based on gradients of c of neighbour LES cells

        print('Computing LES gradients on LES mesh ...')

        # create empty array
        self.grad_c_LES = np.zeros([self.Nx, self.Nx, self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(self.filter_width, self.Nx - self.filter_width):
            for m in range(self.filter_width, self.Nx - self.filter_width):
                for n in range(self.filter_width, self.Nx - self.filter_width):
                    this_LES_gradX = (self.c_filtered[l + self.filter_width, m, n] - self.c_filtered[l - self.filter_width, m, n]) / (2 * self.delta_x * self.filter_width)
                    this_LES_gradY = (self.c_filtered[l, m + self.filter_width, n] - self.c_filtered[l, m - self.filter_width, n]) / (2 * self.delta_x * self.filter_width)
                    this_LES_gradZ = (self.c_filtered[l, m, n + self.filter_width] - self.c_filtered[l, m, n - self.filter_width]) / (2 * self.delta_x * self.filter_width)
                    # compute the magnitude of the gradient
                    this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)

                    self.grad_c_LES[l, m, n] = this_LES_magGrad_c

        return self.grad_c_LES

    #@jit(nopython=True)
    def compute_LES_grad_onLES_reduced(self):
        # computes the flame surface area in the DNS based on gradients of c of neighbour LES cells

        print('Computing LES gradients on LES mesh ...')

        # create empty array
        self.grad_c_LES = np.zeros([self.Nx, self.Nx, self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(self.filter_width, self.Nx - self.filter_width):
            for m in range(self.filter_width, self.Nx - self.filter_width):
                for n in range(self.filter_width, self.Nx - self.filter_width):
                    this_LES_gradX = (self.c_filtered_reduced[l + self.filter_width, m, n] - self.c_filtered_reduced[l - self.filter_width, m, n]) / (2 * self.delta_x * self.filter_width)
                    this_LES_gradY = (self.c_filtered_reduced[l, m + self.filter_width, n] - self.c_filtered_reduced[l, m - self.filter_width, n]) / (2 * self.delta_x * self.filter_width)
                    this_LES_gradZ = (self.c_filtered_reduced[l, m, n + self.filter_width] - self.c_filtered_reduced[l, m, n - self.filter_width]) / (2 * self.delta_x * self.filter_width)
                    # compute the magnitude of the gradient
                    this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)

                    self.grad_c_LES[l, m, n] = this_LES_magGrad_c

        return self.grad_c_LES


    def compute_isoArea(self,c_iso):
        print('Computing the surface for c_iso: ', c_iso)
        # print('Currently in timing test mode!')

        half_filter = int(self.filter_width/2)

        # reference area of planar flame
        A_planar = (self.filter_width - 1)**2

        iterpoints = (self.Nx)**3
        # progress bar
        bar = ChargingBar('Processing', max=iterpoints)

        isoArea_coefficient = np.zeros((self.Nx,self.Nx,self.Nx))

        for l in range(half_filter, self.Nx - half_filter, self.every_nth):
            for m in range(half_filter, self.Nx - half_filter, self.every_nth):
                for n in range(half_filter, self.Nx - half_filter, self.every_nth):

                    this_LES_box = (self.c_data_np[l-half_filter : l+half_filter,
                                                  m-half_filter : m+half_filter,
                                                  n-half_filter : n+half_filter])

                    ###################################
                    # use c_bar for small Delta_LES
                    #print('Small Filter: c_iso = c_bar ')
                    #if self.filter_width < 16:
                    c_iso = np.mean(this_LES_box)
                    ###################################

                    # this works only if the c_iso value is contained in my array
                    # -> check if array contains values above AND below iso value
                    if np.any(np.where(this_LES_box < c_iso)) and np.any(np.any(np.where(this_LES_box > c_iso))):
                        #start = time.time()
                        verts, faces = measure.marching_cubes_classic(this_LES_box, c_iso)
                        iso_area = measure.mesh_surface_area(verts=verts, faces=faces)
                        #end=time.time()
                        #print(' Time marching cube: ', end-start)
                        #iso_area = 0
                    else:
                        iso_area = 0

                    isoArea_coefficient[l, m, n] = iso_area / A_planar

                    # iterbar
                    bar.next()

        bar.finish()

        return isoArea_coefficient

    def compute_isoArea_dynamic(self):
        print('Computing the surface for c_iso based on c_bar ')
        # print('Currently in timing test mode!')

        half_filter = int(self.filter_width/2)

        # reference area of planar flame
        A_planar = (self.filter_width - 1)**2

        iterpoints = (self.Nx)**3
        # progress bar
        bar = ChargingBar('Processing', max=iterpoints)

        isoArea_coefficient = np.zeros((self.Nx,self.Nx,self.Nx))

        for l in range(half_filter, self.Nx - half_filter, self.every_nth):
            for m in range(half_filter, self.Nx - half_filter, self.every_nth):
                for n in range(half_filter, self.Nx - half_filter, self.every_nth):

                    this_LES_box = (self.c_data_np[l-half_filter : l+half_filter,
                                                  m-half_filter : m+half_filter,
                                                  n-half_filter : n+half_filter])

                    # compute c_bar of current LES box
                    this_c_bar = np.mean(this_LES_box)
                    c_iso = this_c_bar
                    print('c_iso: %f' % c_iso)
                    # this works only if the c_iso value is contained in my array
                    # -> check if array contains values above AND below iso value
                    if np.any(np.where(this_LES_box < c_iso)) and np.any(np.any(np.where(this_LES_box > c_iso))):
                        #start = time.time()
                        verts, faces = measure.marching_cubes_classic(this_LES_box, c_iso)
                        iso_area = measure.mesh_surface_area(verts=verts, faces=faces)
                        #end=time.time()
                        #print(' Time marching cube: ', end-start)
                        #iso_area = 0
                    else:
                        iso_area = 0

                    if iso_area / A_planar < 1:
                        isoArea_coefficient[l,m,n] = 0
                    else:
                        isoArea_coefficient[l, m, n] = iso_area / A_planar

                    # iterbar
                    bar.next()

        bar.finish()

        return isoArea_coefficient

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # this is for the parallel approach with joblib
    def compute_isoArea_parallel(self,c_iso):
        print('Computing the surface for c_iso: ', c_iso)

        half_filter = int(self.filter_width/2)

        DNS_range = range(half_filter, self.Nx - half_filter)

        isoArea_coefficient = np.zeros((self.Nx,self.Nx,self.Nx))

        isoArea_list = Parallel(n_jobs=4)(delayed(self.compute_this_LES_box)(l,m,n, half_filter,c_iso,isoArea_coefficient)
                           for n in DNS_range
                           for m in DNS_range
                           for l in DNS_range)

        # reshape isoArea_list into 3D np.array
        isoArea_coefficient = np.array(isoArea_list).reshape(self.Nx,self.Nx,self.Nx)

        return isoArea_coefficient


    def compute_this_LES_box(self,l,m,n, half_filter,c_iso,isoArea_coefficient):

        this_LES_box = (self.c_data_np[l - half_filter: l + half_filter,
                        m - half_filter: m + half_filter,
                        n - half_filter: n + half_filter])

        # this works only if the c_iso value is contained in my array
        # -> check if array contains values above AND below iso value
        try: #if np.any(np.where(this_LES_box < c_iso)) and np.any(np.any(np.where(this_LES_box > c_iso))):
            verts, faces = measure.marching_cubes_classic(this_LES_box, c_iso)
            iso_area = measure.mesh_surface_area(verts=verts, faces=faces)
        except ValueError: #else:
            iso_area = 0

        #isoArea_coefficient[l, m, n] = iso_area / (self.filter_width - 1) ** 2

        return iso_area / (self.filter_width - 1) ** 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def compute_filter_DNS_grad(self):
    # compute filtered DNS reaction rate
         # create empty array
         #grad_DNS_filtered = np.zeros([self.Nx, self.Nx, self.Nx])

        # compute dask delayed object
        grad_c_DNS = self.compute_DNS_grad()

        grad_DNS_filtered = self.apply_filter(grad_c_DNS)

        return grad_DNS_filtered

    def compute_filter_DNS_grad_4thO(self):
         # compute filtered DNS reaction rate
         # create empty array
         #grad_DNS_filtered = np.zeros([self.Nx, self.Nx, self.Nx])

        # compute dask delayed object
        grad_c_DNS = self.compute_DNS_grad_4thO()

        grad_DNS_filtered = self.apply_filter(grad_c_DNS)

        return grad_DNS_filtered


    def compute_filter_DNS_grad_reduced(self):
    # compute filtered DNS reaction rate

        # compute dask delayed object
        grad_c_DNS = self.compute_DNS_grad_reduced()

        grad_DNS_filtered = self.apply_filter(grad_c_DNS)

        return grad_DNS_filtered


    def compute_RR_DNS(self):

        c_data_np_vector = self.c_data_np.reshape(self.Nx**3)

        # according to Pfitzner implementation
        exponent = - self.beta*(1 - c_data_np_vector) / (1 - self.alpha*(1 - c_data_np_vector))

        this_RR_reshape_DNS_Pfitz  = 18.97 * ((1 - self.alpha * (1 - c_data_np_vector))) ** (-1) \
                                     * (1 - c_data_np_vector) * np.exp(exponent)

        RR_DNS = this_RR_reshape_DNS_Pfitz.reshape(self.Nx,self.Nx,self.Nx)

        return RR_DNS


    def filter_RR_DNS(self):
    # compute filtered DNS reaction rate

        RR_DNS_filtered = self.apply_filter(self.omega_DNS)

        return RR_DNS_filtered


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
        EVTL MUSS FILTERWIDTH NOCH SKALIERT WERDEN! -> Sollte stimmen!
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
        # self.c_filtered.reshape(self.Nx**3) = c_bar in der ganzen domain als vector
        this_s = self.compute_s(self.c_filtered.reshape(self.Nx**3))
        this_delta_0 = self.compute_delta_0(this_s)

        self.c_minus = (np.exp(self.c_filtered.reshape(self.Nx**3)* this_delta_0 * self.Delta_LES) - 1) / \
                       (np.exp(this_delta_0 * self.Delta_LES) - 1)


    # Analytical c_minus (Eq. 35)
    def compute_c_minus_analytical(self):
        # generate a dummy c_minus vector
        c_minus_dummy = np.linspace(0,0.99999,100000)

        # compute c_plus
        this_xi_m = self.compute_xi_m(c_minus_dummy)
        xi_plus_Delta = this_xi_m + self.Delta_LES
        c_plus_dummy = self.compute_c_m(xi_plus_Delta)

        # compute upper bound profile based on c_bar_dumma
        upper_bound = self.I_1(c_plus_dummy)
        lower_bound = self.I_1(c_minus_dummy)

        c_bar_dummy = (upper_bound - lower_bound) # /self.Delta_LES # nicht sicher Delta_LES

        # interpolate from c_minus_profile to correct c_minus based on c_filtered
        f_c_minus = interpolate.interp1d(c_bar_dummy, c_minus_dummy,fill_value="extrapolate")
        f_c_plus = interpolate.interp1d(c_bar_dummy, c_plus_dummy, fill_value="extrapolate")

        # update c_minus and c_plus
        self.c_minus = f_c_minus(self.c_filtered.reshape(self.Nx**3))
        self.c_plus = f_c_plus(self.c_filtered.reshape(self.Nx ** 3))


    def I_1(self,c):
        '''
        :param c:
        :return: Hypergeometric function (Eq. 35)
        '''
        return c / self.Delta_LES * special.hyp2f1(1, 1 / self.m, 1 + 1 / self.m, c ** self.m)

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
        print('Computing omega model ...')

        omega_cbar = ((self.c_plus ** (self.m + 1) - self.c_minus ** (self.m + 1)) / self.Delta_LES)

        # print('Writing omega Pfitzner model to file ...')
        # filename = join(self.case, 'omega_model_' + str(self.filter_width) + '.csv')
        # pd.DataFrame(data=omega_cbar.reshape(self.Nx**3),columns=['omega_model']).to_csv(filename)

        return omega_cbar.reshape(self.Nx,self.Nx,self.Nx)


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

        print('Computing omega DNS ...')

        exponent = - (self.beta * (1 - c)) / (1 - self.alpha * (1 - c))
        Eigenval = 18.97  # beta**2 / 2 + beta*(3*alpha - 1.344)
        # --> Eigenval ist wahrscheinlich falsch!

        # om_Klein = self.bfact*self.rho_bar*(1-c)*np.exp(exponent)
        om_Pfitzner = Eigenval * ((1 - self.alpha * (1 - c))) ** (-1) * (1 - c) * np.exp(exponent)

        return om_Pfitzner


    def compute_Pfitzner_model(self):
        '''
        computes the model values in sequential manner
        :param self.omega_model_cbar: is the modeled omega from the laminar planar flame
        :param self.omega_DNS: is the (real) omega field from the DNS data
        :param self.omega_DNS_filtered: is the filtered omega field from DNS data; this is the bench mark to compare with
        '''

        # switch between the computation modes
        if self.c_analytical is True:
            self.compute_c_minus_analytical()
        else:
            self.compute_c_minus()
            self.compute_c_plus()

        self.omega_model_cbar = self.compute_model_omega_bar()

        #self.omega_model_cbar = self.model_omega(self.c_filtered.reshape(self.Nx**3))
        self.omega_DNS = self.analytical_omega(self.c_data_np.reshape(self.Nx**3))

        if len(self.omega_DNS.shape) == 1:
            self.omega_DNS = self.omega_DNS.reshape(self.Nx,self.Nx,self.Nx)

        # filter the DNS reaction rate
        print('Filtering omega DNS ...')

        self.omega_DNS_filtered = self.apply_filter(self.omega_DNS)



# NEW CLASS FOR THE CLUSTER
#TODO: Evtl wieder löschen...

class data_binning_cluster(data_binning_PDF):

    def run_analysis_wrinkling(self,filter_width ,filter_type, c_analytical=False, Parallel=False, every_nth=1):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :param Parallel: use False
        :param every_nth: every nth DNS point to compute the isoArea
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        # joblib parallel computing of c_iso
        self.Parallel = Parallel

        self.every_nth = int(every_nth)

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        # filter the c and rho field
        print('Filtering c field ...')
        self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

        # # reduce c for computation of conditioned wrinkling factor
        # self.reduce_c(c_min=0.75,c_max=0.85)
        # self.c_filtered_reduced = self.apply_filter(self.c_data_reduced_np)

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)

        # Set the Gauss kernel
        self.set_gaussian_kernel()

        # compute the wrinkling factor
        self.get_wrinkling()
        self.compute_Pfitzner_model()

        #c_bins = self.compute_c_binning(c_low=0.8,c_high=0.9)

        start = time.time()
        if self.Parallel is True:
            isoArea_coefficient = self.compute_isoArea_parallel(c_iso=0.85)
        else:
            isoArea_coefficient = self.compute_isoArea(c_iso=0.85)

        end=time.time()
        print('computation of c_iso took %i sec: ' % int(end - start))

        # write the filtered data of the whole DNS cube only if every data point is filtered. No sparse data...(every_nth > 1)
        if self.every_nth == 1:
            # write the filtered omega and omega_model * isoArea to file
            print('writing omega DNS filtered and omega_model x isoArea to file ...')
            filename = join(self.case, 'filtered_data','omega_filtered_modeled_' + str(self.filter_width) +'_nth'+ str(self.every_nth) + '.csv')

            om_iso = self.omega_model_cbar*isoArea_coefficient
            om_wrinkl = self.omega_model_cbar*self.wrinkling_factor

            pd.DataFrame(data=np.hstack([self.omega_DNS.reshape(self.Nx**3,1),
                               self.omega_DNS_filtered.reshape(self.Nx**3,1),
                               om_iso.reshape(self.Nx**3,1),
                               om_wrinkl.reshape(self.Nx**3,1),
                               self.c_filtered.reshape(self.Nx ** 3, 1)]),
                               columns=['omega_DNS',
                                        'omega_filtered',
                                        'omega_model_by_isoArea',
                                        'omega_model_by_wrinkling',
                                        'c_bar']).to_csv(filename)


        # creat dask array and reshape all data
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx**3,1),
                                   self.wrinkling_factor.reshape(self.Nx**3,1),
                                   isoArea_coefficient.reshape(self.Nx**3,1),
                                   # self.wrinkling_factor_LES.reshape(self.Nx ** 3, 1),
                                   # self.wrinkling_factor_reduced.reshape(self.Nx ** 3, 1),
                                   # self.wrinkling_factor_LES_reduced.reshape(self.Nx ** 3, 1),
                                   self.omega_model_cbar.reshape(self.Nx**3,1),
                                   self.omega_DNS_filtered.reshape(self.Nx**3,1),
                                   #self.omega_LES_noModel.reshape(self.Nx**3,1),
                                   self.c_plus.reshape(self.Nx**3,1),
                                   self.c_minus.reshape(self.Nx**3,1)])


        if self.c_analytical is True:
            # write data to csv file
            filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_analytical.csv')
        else:
            # write data to csv file
            filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '.csv')


        self.dataArray_dd = dd.io.from_dask_array(dataArray_da,
                                             columns=['c_bar',
                                                      'wrinkling',
                                                      'isoArea',
                                                      # 'wrinkling_LES',
                                                      # 'wrinkling_reduced',
                                                      # 'wrinkling_LES_reduced',
                                                      'omega_model',
                                                      'omega_DNS_filtered',
                                                      #'omega_cbar',
                                                      'c_plus',
                                                      'c_minus'])

        # filter the data set and remove unecessary entries
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] > 0.01]
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] < 0.99]

        if self.case is 'planar_flame_test':
            self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['wrinkling'] < 1.1]

        # self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['wrinkling'] > 0.99]
        # self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['isoArea'] >= 1.0]
        print('isoArea < 1 is included!')

        # this is to reduce the storage size
        #self.dataArray_dd = self.dataArray_dd.sample(frac=0.3)

        print('Computing data array ...')
        self.dataArray_df = self.dataArray_dd.compute()

        print('Writing output to csv ...')
        self.dataArray_df.to_csv(filename,index=False)
        print('Data has been written.\n\n')

    def plot_histograms(self, c_tilde, this_rho_c_reshape, this_rho_reshape, this_RR_reshape_DNS):
        return NotImplementedError

    def plot_histograms_intervals(self,c_tilde,this_rho_c_reshape,this_rho_reshape,c_bar,this_RR_reshape_DNS,wrinkling=1):
        return NotImplementedError





###################################################################
# NEW CLASS WITH PFITZNERS WRINKLING FACTOR

class data_binning_dirac(data_binning_PDF):
    # new implementation with numerical delta dirac function

    def __init__(self, case, bins, eps_factor=66, c_iso_values=[0.5]):
        # extend super class with additional input parameters
        super(data_binning_dirac, self).__init__(case, bins)
        self.c_iso_values = c_iso_values
        self.eps_factor = eps_factor
        # the convolution of dirac x grad_c at all relevant iso values and LES filtered will be stored here
        self.Xi_iso_filtered = np.zeros((self.Nx,self.Nx,self.Nx,len(self.c_iso_values)))

        print('You are using the Dirac version...')

    def compute_phi_c(self,c_iso):
        # computes the difference between abs(c(x)-c_iso)
        # see Pfitzner notes Eq. 3

        return abs(self.c_data_np - c_iso)

    def compute_dirac_cos(self,c_phi):
        '''
        :param c_phi: c(x) - c_iso
        :param m: scaling factor (Zahedi paper)
        :return: numerical dirac delta function
        '''
        eps = self.eps_factor*self.delta_x

        X = c_phi/eps

        # transform to vector
        X_vec = X.reshape(self.Nx**3)

        dirac_vec = np.zeros(len(X_vec))

        # Fallunterscheidung für X < 0
        for id, x in enumerate(X_vec):
            if x < 1:
                dirac_vec[id] =1/(2*eps) * (1 + np.cos(np.pi * x)) * self.delta_x
                #print('dirac_vec: ',dirac_vec[id])

        dirac_array = dirac_vec.reshape(self.Nx,self.Nx,self.Nx)

        return dirac_array


    def compute_Xi_iso_dirac(self):

        # check if self.grad_c_DNS was computed, if not -> compute it
        if type(self.grad_c_DNS) is None:
            self.grad_c_DNS=self.compute_DNS_grad_4thO()

        # loop over the different c_iso values
        for id, c_iso in enumerate(self.c_iso_values):

            c_phi = self.compute_phi_c(c_iso)
            dirac = self.compute_dirac_cos(c_phi)
            self.dirac_times_grad_c = (dirac * self.grad_c_DNS).reshape(self.Nx,self.Nx,self.Nx)
            print('dirac_imes_grad_c: ', self.dirac_times_grad_c)

            # check if integral is 1 for arbitrary
            # Line integral
            print('line integrals over dirac_times_grad(c):')
            # print(np.trapz(dirac_times_grad_c[:, 250, 250]))
            # print(np.trapz(dirac_times_grad_c[250, :, 250]))
            print(np.trapz(self.dirac_times_grad_c[250, 250, :]))

            # save the line
            output_df = pd.DataFrame(data=np.vstack([self.dirac_times_grad_c[250, 250, :],
                                                    c_phi[250,250,:],
                                                    dirac[250, 250, :],
                                                    self.c_data_np[250,250,:]]).transpose(),
                                     columns=['dirac_grad_c','c_phi','dirac','c'])
            output_df.to_csv(join(self.case, '1D_data_cube.csv'))

            if c_iso == 0.5:
                #self.dirac_times_grad_c_085 = dirac_times_grad_c
                self.grad_c_05 = self.grad_c_DNS.reshape(self.Nx,self.Nx,self.Nx)
                self.dirac_05 = dirac.reshape(self.Nx,self.Nx,self.Nx)

            # TODO: check if that is correct!
            dirac_LES_sums = self.compute_LES_cell_sum(self.dirac_times_grad_c)
            self.Xi_iso_filtered[:, :, :, id] = dirac_LES_sums / self.filter_width**2

            #apply TOPHAT filter to dirac_times_grad_c --> Obtain surface
            # Check Eq. 6 from Pfitzner notes (Generalized wrinkling factor)
            # print('Filtering for Xi_iso at c_iso: %f' % c_iso)

            # # stimmt vermutlich nicht...
            # self.Xi_iso_filtered[:,:,:,id] = self.apply_filter(self.dirac_times_grad_c) / self.Delta_LES**2
            # # TTODO: check if self.Delta_LES is correct model parameter here

    # overrides main method
    def run_analysis_dirac(self,filter_width ,filter_type, c_analytical=False, Parallel=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :param Parallel: use False
        :param every_nth: every nth DNS point to compute the isoArea
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        # joblib parallel computing of c_iso
        self.Parallel = Parallel

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        # filter the c and rho field
        print('Filtering c field ...')
        self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

        # # reduce c for computation of conditioned wrinkling factor
        # self.reduce_c(c_min=0.75,c_max=0.85)
        # self.c_filtered_reduced = self.apply_filter(self.c_data_reduced_np)

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)

        # Set the Gauss kernel
        self.set_gaussian_kernel()

        # compute the wrinkling factor: NOT NEEDED here!
        # self.get_wrinkling()

        # compute abs(grad(c)) on the whole DNS domaine
        self.grad_c_DNS = self.compute_DNS_grad_4thO()

        # compute omega based on pfitzner
        self.compute_Pfitzner_model()

        #c_bins = self.compute_c_binning(c_low=0.8,c_high=0.9)

        # compute Xi iso surface area for all c_iso values
        self.compute_Xi_iso_dirac()

        # creat dask array and reshape all data
        # a bit nasty for list in list as of variable c_iso values
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx**3,1),
                                    self.Xi_iso_filtered[:,:,:,0].reshape(self.Nx**3,1),
                                    # self.Xi_iso_filtered[:, :, :,1].reshape(self.Nx**3,1),
                                    # self.Xi_iso_filtered[:, :, :, 2].reshape(self.Nx**3,1),
                                    # self.Xi_iso_filtered[:, :, :, 3].reshape(self.Nx**3,1),
                                    self.omega_model_cbar.reshape(self.Nx**3,1),
                                    self.omega_DNS_filtered.reshape(self.Nx**3,1),
                                    #self.c_plus.reshape(self.Nx**3,1),
                                    #self.c_minus.reshape(self.Nx**3,1),
                                    self.grad_c_05.reshape(self.Nx**3,1),
                                    self.dirac_05.reshape(self.Nx ** 3, 1)
                                  ])

        filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_dirac.csv')

        self.dataArray_dd = dd.io.from_dask_array(dataArray_da,
                                             columns=['c_bar',
                                                      'Xi_iso_0.5',
                                                      # 'Xi_iso_0.75',
                                                      # 'Xi_iso_0.85',
                                                      # 'Xi_iso_0.95',
                                                      'omega_model',
                                                      'omega_DNS_filtered',
                                                      #'c_plus',
                                                      #'c_minus',
                                                      'grad_DNS_c_05',
                                                      'dirac_05'])

        # filter the data set and remove unecessary entries
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] > 0.01]
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] < 0.99]


        # remove all Xi_iso < 1e-2 from the stored data set -> less memory
        self.dataArray_dd = self.dataArray_dd[
                                                (self.dataArray_dd['Xi_iso_0.5'] > 1e-2) ]#&
                                               # (self.dataArray_dd['Xi_iso_0.75'] > 1e-2) &
                                               # (self.dataArray_dd['Xi_iso_0.85'] > 1e-2) &
                                               # (self.dataArray_dd['Xi_iso_0.95'] > 1e-2) ]
        print('Xi_iso < 1 are included!')

        print('Computing data array ...')
        self.dataArray_df = self.dataArray_dd.sample(frac=0.2).compute()

        print('Writing output to csv ...')
        self.dataArray_df.to_csv(filename,index=False)
        print('Data has been written.\n\n')


    def compute_LES_cell_sum(self,input_array):
        # get the sum of values inside an LES cell
        print('computing cell sum')

        try:
            assert len(input_array.shape) == 3
        except AssertionError:
            print('input array must be 3D!')

        output_array = copy.copy(input_array)

        output_array *= 0.0                 # set output array to zero

        half_filter = int(self.filter_width / 2)

        for l in range(half_filter, self.Nx - half_filter, 1):
            for m in range(half_filter, self.Nx - half_filter, 1):
                for n in range(half_filter, self.Nx - half_filter, 1):

                    this_LES_box = (input_array[l - half_filter: l + half_filter,
                                    m - half_filter: m + half_filter,
                                    n - half_filter: n + half_filter])

                    # compute c_bar of current LES box
                    output_array[l,m,n] = this_LES_box.sum()

        return output_array


    def run_analysis_wrinkling(self,filter_width ,filter_type, c_analytical=False, Parallel=False, every_nth=1):
        print('Use run_analysis_dirac instead')
        return NotImplementedError

    def plot_histograms(self, c_tilde, this_rho_c_reshape, this_rho_reshape, this_RR_reshape_DNS):
        return NotImplementedError

    def plot_histograms_intervals(self,c_tilde,this_rho_c_reshape,this_rho_reshape,c_bar,this_RR_reshape_DNS,wrinkling=1):
        return NotImplementedError




###################################################################
# NEW CLASS WITH PFITZNERS WRINKLING FACTOR
# THEREs a difference between xi and Xi!!! see the paper...

class data_binning_dirac_xi(data_binning_PDF):
    # new implementation with numerical delta dirac function

    def __init__(self, case, bins, eps_factor=100, c_iso_values=[0.85]):
        # extend super class with additional input parameters
        super(data_binning_dirac_xi, self).__init__(case, bins)
        self.c_iso_values = c_iso_values
        self.eps_factor = eps_factor
        # the convolution of dirac x grad_c at all relevant iso values and LES filtered will be stored here
        self.Xi_iso_filtered = np.zeros((self.Nx,self.Nx,self.Nx,len(self.c_iso_values)))

        self.xi_np = None        # to be filled
        self.xi_iso_values = None   # converted c_iso_values into xi-space
        self.grad_xi_DNS = None     # xi-Field gradients on DNS mesh
        self.dirac_times_grad_xi = None

        print('You are using the Dirac version...')

    def convert_to_xi(self):
        # converts the c-field to the xi field (Eq. 13, Pfitzner)

        c_clipped = self.c_data_np*0.9999 + 1e-5

        self.xi_np = 1/self.m * np.log(c_clipped**self.m/ (1 - c_clipped**self.m) )
        self.xi_iso_values = [1/self.m * np.log(c**self.m/ (1 - c**self.m) ) for c in self.c_iso_values] #is a list


    def compute_phi_xi(self,xi_iso):
        # computes the difference between abs(c(x)-c_iso)
        # see Pfitzner notes Eq. 3

        return abs(self.xi_np - xi_iso)

    def compute_dirac_cos(self,xi_phi):
        '''
        EVERYTHING IS COMPUTED IN xi SPACE!
        :param xi_phi: xi(x) - xi_iso
        :param m: scaling factor (Zahedi paper)
        :return: numerical dirac delta function for whole field
        '''
        eps = self.eps_factor*self.delta_x

        X = xi_phi/eps

        # transform to vector
        X_vec = X.reshape(self.Nx**3)

        dirac_vec = np.zeros(len(X_vec))

        # Fallunterscheidung für X < 0
        for id, x in enumerate(X_vec):
            if x < 1:
                dirac_vec[id] =1/(2*eps) * (1 + np.cos(np.pi * x)) * self.delta_x
                #print('dirac_vec: ',dirac_vec[id])

        dirac_array = dirac_vec.reshape(self.Nx,self.Nx,self.Nx)

        return dirac_array


    def compute_Xi_iso_dirac_xi(self):

        # check if self.grad_c_DNS was computed, if not -> compute it
        if type(self.grad_xi_DNS) is None:
            self.compute_DNS_grad_xi()

        # loop over the different c_iso values
        for id, xi_iso in enumerate(self.xi_iso_values):

            xi_phi = self.compute_phi_xi(xi_iso)
            dirac_xi = self.compute_dirac_cos(xi_phi)
            self.dirac_times_grad_xi = (dirac_xi * self.grad_xi_DNS).reshape(self.Nx,self.Nx,self.Nx)
            #print('dirac_times_grad_xi: ', self.dirac_times_grad_xi)

            # check if integral is 1 for arbitrary
            # Line integral
            print('line integrals over dirac_times_grad(c):')
            # print(np.trapz(dirac_times_grad_c[:, 250, 250]))
            # print(np.trapz(dirac_times_grad_c[250, :, 250]))
            print(np.trapz(self.dirac_times_grad_xi[250, 250, :]))

            # save the line
            output_df = pd.DataFrame(data=np.vstack([self.dirac_times_grad_xi[250, 250, :],
                                                    xi_phi[250,250,:],
                                                    dirac_xi[250, 250, :],
                                                    self.xi_np[250,250,:]]).transpose(),
                                     columns=['dirac_grad_xi','xi_phi','dirac_xi','xi'])
            output_df.to_csv(join(self.case, '1D_data_cube.csv'))

            #if xi_iso == 0.85:
            #self.dirac_times_grad_c_085 = dirac_times_grad_c
            self.grad_c_05 = self.grad_xi_DNS.reshape(self.Nx,self.Nx,self.Nx)
            self.dirac_05 = dirac_xi.reshape(self.Nx,self.Nx,self.Nx)

            # TODO: check if that is correct!
            dirac_LES_sums = self.compute_LES_cell_sum(self.dirac_times_grad_xi)
            self.Xi_iso_filtered[:, :, :, id] = dirac_LES_sums / self.filter_width**2

    # overrides main method
    def run_analysis_dirac(self,filter_width ,filter_type, c_analytical=False, Parallel=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :param Parallel: use False
        :param every_nth: every nth DNS point to compute the isoArea
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        # joblib parallel computing of c_iso
        self.Parallel = Parallel

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        # filter the c and rho field
        print('Filtering c field ...')
        self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

        # # reduce c for computation of conditioned wrinkling factor
        # self.reduce_c(c_min=0.75,c_max=0.85)
        # self.c_filtered_reduced = self.apply_filter(self.c_data_reduced_np)

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)

        # Set the Gauss kernel
        self.set_gaussian_kernel()

        # compute the wrinkling factor: NOT NEEDED here!
        # self.get_wrinkling()

        # compute abs(grad(c)) on the whole DNS domain
        self.grad_xi_DNS = self.compute_DNS_grad_xi()

        # compute omega based on pfitzner
        self.compute_Pfitzner_model()

        #c_bins = self.compute_c_binning(c_low=0.8,c_high=0.9)

        # compute Xi iso surface area for all c_iso values
        self.compute_Xi_iso_dirac_xi()

        # creat dask array and reshape all data
        # a bit nasty for list in list as of variable c_iso values
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx**3,1),
                                    self.Xi_iso_filtered[:,:,:,0].reshape(self.Nx**3,1),
                                    # self.Xi_iso_filtered[:, :, :,1].reshape(self.Nx**3,1),
                                    # self.Xi_iso_filtered[:, :, :, 2].reshape(self.Nx**3,1),
                                    # self.Xi_iso_filtered[:, :, :, 3].reshape(self.Nx**3,1),
                                    self.omega_model_cbar.reshape(self.Nx**3,1),
                                    self.omega_DNS_filtered.reshape(self.Nx**3,1),
                                    #self.c_plus.reshape(self.Nx**3,1),
                                    #self.c_minus.reshape(self.Nx**3,1),
                                    self.grad_c_05.reshape(self.Nx**3,1),
                                    self.dirac_05.reshape(self.Nx ** 3, 1)
                                  ])

        filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_dirac_xi.csv')

        self.dataArray_dd = dd.io.from_dask_array(dataArray_da,
                                             columns=['c_bar',
                                                      'Xi_iso_0.5',
                                                      # 'Xi_iso_0.75',
                                                      # 'Xi_iso_0.85',
                                                      # 'Xi_iso_0.95',
                                                      'omega_model',
                                                      'omega_DNS_filtered',
                                                      #'c_plus',
                                                      #'c_minus',
                                                      'grad_DNS_c_05',
                                                      'dirac_05'])

        # filter the data set and remove unecessary entries
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] > 0.01]
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] < 0.99]


        # remove all Xi_iso < 1e-2 from the stored data set -> less memory
        self.dataArray_dd = self.dataArray_dd[
                                                (self.dataArray_dd['Xi_iso_0.5'] > 1e-2) ]#&
                                               # (self.dataArray_dd['Xi_iso_0.75'] > 1e-2) &
                                               # (self.dataArray_dd['Xi_iso_0.85'] > 1e-2) &
                                               # (self.dataArray_dd['Xi_iso_0.95'] > 1e-2) ]
        print('Xi_iso < 1 are included!')

        print('Computing data array ...')
        self.dataArray_df = self.dataArray_dd.sample(frac=0.2).compute()

        print('Writing output to csv ...')
        self.dataArray_df.to_csv(filename,index=False)
        print('Data has been written.\n\n')


    def compute_LES_cell_sum(self,input_array):
        # get the sum of values inside an LES cell
        print('computing cell sum...')

        try:
            assert len(input_array.shape) == 3
        except AssertionError:
            print('input array must be 3D!')

        output_array = copy.copy(input_array)

        output_array *= 0.0                 # set output array to zero

        half_filter = int(self.filter_width / 2)

        for l in range(half_filter, self.Nx - half_filter, 1):
            for m in range(half_filter, self.Nx - half_filter, 1):
                for n in range(half_filter, self.Nx - half_filter, 1):

                    this_LES_box = (input_array[l - half_filter: l + half_filter,
                                    m - half_filter: m + half_filter,
                                    n - half_filter: n + half_filter])

                    # compute c_bar of current LES box
                    output_array[l,m,n] = this_LES_box.sum()

        return output_array

    def compute_DNS_grad_xi(self):
        # computes the flame surface area in the DNS based on gradients of xi of neighbour cells
        # magnitude of the gradient: abs(grad(c))

        print('Computing DNS gradients in xi space...')

        # check if self.xi_np is filled
        if self.xi_np is None:
            self.convert_to_xi()

        # create empty array
        grad_xi_DNS = np.zeros([self.Nx,self.Nx,self.Nx])

        # compute gradients from the boundaries away ...
        for l in range(1,self.Nx-1):
            for m in range(1,self.Nx-1):
                for n in range(1,self.Nx-1):
                    this_DNS_gradX = (self.xi_np[l+1, m, n] - self.xi_np[l-1,m, n])/(2 * self.delta_x)
                    this_DNS_gradY = (self.xi_np[l, m+1, n] - self.xi_np[l, m-1, n]) / (2 * self.delta_x)
                    this_DNS_gradZ = (self.xi_np[l, m, n+1] - self.xi_np[l, m, n-1]) / (2 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_DNS_magGrad_c = np.sqrt(this_DNS_gradX**2 + this_DNS_gradY**2 + this_DNS_gradZ**2)

                    grad_xi_DNS[l,m,n] = this_DNS_magGrad_c

        return grad_xi_DNS


    def run_analysis_wrinkling(self,filter_width ,filter_type, c_analytical=False, Parallel=False, every_nth=1):
        print('Use run_analysis_dirac instead')
        return NotImplementedError

    def plot_histograms(self, c_tilde, this_rho_c_reshape, this_rho_reshape, this_RR_reshape_DNS):
        return NotImplementedError

    def plot_histograms_intervals(self,c_tilde,this_rho_c_reshape,this_rho_reshape,c_bar,this_RR_reshape_DNS,wrinkling=1):
        return NotImplementedError




###################################################################
# NEW CLASS WITH PFITZNERS FSD FORMULATION FOR GENERALIZED WRINKLING FACTOR
# SEE report from Pfitzner 12.2.2020

#TODO: remove!
# This version is OUTDATED
# Check if it is a parent Class??

class data_binning_dirac_FSD(data_binning_dirac_xi):
    # new implementation with numerical delta dirac function

    def __init__(self,case, bins, eps_factor=100,c_iso_values=[0.01,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99]):
        # extend super class with additional input parameters
        super(data_binning_dirac_FSD, self).__init__(case, bins, eps_factor,c_iso_values)
        # self.c_iso_values = c_iso_values
        # self.eps_factor = eps_factor
        # the convolution of dirac x grad_c at all relevant iso values and LES filtered will be stored here
        #self.Xi_iso_filtered = np.zeros((self.Nx,self.Nx,self.Nx,len(self.c_iso_values)))

        # self.xi_np = None        # to be filled
        # self.xi_iso_values = None
        # self.grad_xi_DNS = None
        # self.dirac_times_grad_xi = None

        # number of c_iso slices
        self.N_c_iso = len(self.c_iso_values)

        # this is a 4D array to store all dirac values for the different c_iso values
        self.dirac_xi_fields = np.zeros([self.N_c_iso,self.Nx,self.Nx,self.Nx])

        # set up a new omega_bar field to store the exact solution from the pdf...no real name yet
        self.omega_model_exact = np.zeros([self.Nx,self.Nx,self.Nx])

        #print('You are using the Dirac version...')
        print('You are using the new FSD routine...')

    def compute_dirac_xi_iso_fields(self):

        # compute the xi field
        # self.xi_np and self.xi_iso_values (from self.c_iso_values)
        self.convert_to_xi()

        # loop over the different c_iso values
        for id, xi_iso in enumerate(self.xi_iso_values):

            print('Computing delta_dirac in xi-space for c_iso=%f'%self.c_iso_values[id])
            xi_phi = self.compute_phi_xi(xi_iso)
            dirac_xi = self.compute_dirac_cos(xi_phi)

            #write each individual dirac_xi field into the storage array
            self.dirac_xi_fields[id,:,:,:] = dirac_xi


    def compute_omega_FSD(self):
        # omega computation with FSD method according to Pfitzner
        # omega_bar = \int_0^1 (m+1)*c_iso**m 1/Delta_LES**3 \int_cell \delta(\xi(x) - \xi_iso) dx dc_iso

        print('Computing exact omega bar ...')

        self.compute_dirac_xi_iso_fields()

        half_filter = int(self.filter_width / 2)

        # Loop over the cells ...
        print('Looping over the cells')
        for l in range(half_filter,self.Nx-half_filter):
            for m in range(half_filter,self.Nx-half_filter):
                for n in range(half_filter,self.Nx-half_filter):
                    # loop over the individual c_iso/xi_iso values

                    omega_iso_vec = np.zeros(self.N_c_iso)
                    for id, this_c_iso in enumerate(self.c_iso_values):

                        this_xi_dirac_box = self.dirac_xi_fields[id,l-half_filter:l+half_filter, m-half_filter:m+half_filter, n-half_filter:n+half_filter]

                        # get the integral over the box -> sum up, divide by Volume
                        omega_over_grad_c = self.compute_omega_over_grad_c(this_c_iso)

                        #TODO: check if Delta_LES**3 is correct!
                        omega_iso_vec[id] = omega_over_grad_c * 1/self.Delta_LES**3 * this_xi_dirac_box.sum()

                    omega_integrated = simps(omega_iso_vec,self.c_iso_values)   # TODO: Alternativ: xi_iso_values

                    self.omega_model_exact[l,m,n] = omega_integrated


    def compute_omega_over_grad_c(self,c_iso):
        '''
        Computes the analytical expression for \omega(c*)/|dc*/dx|
        has to be computed for each c_iso value indivdually
        :return:
        '''

        return (self.m + 1)* c_iso** self.m


    def run_analysis_dirac_FSD(self,filter_width ,filter_type, c_analytical=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        # filter the c and rho field
        print('Filtering c field ...')
        self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

        # # reduce c for computation of conditioned wrinkling factor
        # self.reduce_c(c_min=0.75,c_max=0.85)
        # self.c_filtered_reduced = self.apply_filter(self.c_data_reduced_np)

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)

        # compute omega based on pfitzner
        self.compute_Pfitzner_model()

        # compute omega_model_exact with FSD from Pfitzner
        self.compute_omega_FSD()

        # creat dask array and reshape all data
        # a bit nasty for list in list as of variable c_iso values
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx**3,1),
                                    self.omega_model_exact.reshape(self.Nx**3,1),
                                    self.omega_DNS_filtered.reshape(self.Nx**3,1)
                                  ])

        filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_FSD_xi.csv')

        self.dataArray_dd = dd.io.from_dask_array(dataArray_da,columns=['c_bar','omega_model_exact','omega_DNS_filtered'])

        # filter the data set and remove unecessary entries
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] > 0.01]
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] < 0.99]


        print('Computing data array ...')
        self.dataArray_df = self.dataArray_dd.sample(frac=0.5).compute()

        print('Writing output to csv ...')
        self.dataArray_df.to_csv(filename,index=False)
        print('Data has been written.\n\n')



###################################################################
# NEW CLASS WITH PFITZNERS FSD FORMULATION FOR GENERALIZED WRINKLING FACTOR
# SEE report from Pfitzner 12.2.2020

# ALTERNATIVE FORMULATION!!!
# IDEA: compute iso-fields, then convolute them omega(c)/dc/dxi, then integrate, then filter!

class data_binning_dirac_FSD_alt(data_binning_dirac_xi):
    # new implementation with numerical delta dirac function

    def __init__(self,case, bins, eps_factor=100,
                 c_iso_values=[0.001, 0.3,0.5,0.55,0.6,0.65,0.7,0.725,0.75,0.775,0.8,0.82,0.84,0.85,0.86,0.88,0.9,0.92,0.94,0.98 ,0.999]):
        # extend super class with additional input parameters
        super(data_binning_dirac_FSD_alt, self).__init__(case, bins, eps_factor,c_iso_values)

        # number of c_iso slices
        self.N_c_iso = len(self.c_iso_values)

        # this is a 4D array to store all dirac values for the different c_iso values
        self.dirac_xi_fields = np.zeros([self.N_c_iso,self.Nx,self.Nx,self.Nx])

        # set up a new omega_bar field to store the exact solution from the pdf...no real name yet
        self.omega_model_exact = np.zeros([self.Nx,self.Nx,self.Nx])

        print('You are using the new alternative FSD routine...')


    def compute_dirac_xi_iso_fields(self):

        # compute the xi field
        # self.xi_np and self.xi_iso_values (from self.c_iso_values)
        self.convert_to_xi()

        # loop over the different c_iso values
        for id, xi_iso in enumerate(self.xi_iso_values):

            print('Computing delta_dirac in xi-space for c_iso=%f'%self.c_iso_values[id])
            xi_phi = self.compute_phi_xi(xi_iso)
            dirac_xi = self.compute_dirac_cos(xi_phi)

            #write each individual dirac_xi field into the storage array
            self.dirac_xi_fields[id,:,:,:] = dirac_xi


    def compute_omega_FSD(self):
        '''
        omega computation with FSD method according to Pfitzner
        omega_bar = \int_0^1 (m+1)*c_iso**m 1/Delta_LES**3 \int_cell \delta(\xi(x) - \xi_iso) dx dc_iso
        :return: NULL
        :param self.omega_model_exact is the FSD omgea
        '''

        print('Computing exact omega bar ...')

        self.compute_dirac_xi_iso_fields()

        # convolute the iso fields wiht omega/dc/dxi
        for id, this_c_iso in enumerate(self.c_iso_values):

            omega_over_grad_c = self.compute_omega_over_grad_c(this_c_iso)
            self.dirac_xi_fields[id,:,:,:] = self.dirac_xi_fields[id,:,:,:] * omega_over_grad_c #ACHTUNG: überschreibt self.dirac_xi_fields!

        # do simpson integration over the whole range of c_iso values
        omega_integrated = simps(self.dirac_xi_fields,self.c_iso_values,axis=0)

        try:
            assert omega_integrated == (self.Nx,self.Nx,self.Nx)
        except AssertionError:
            print('omega_integrant shape', omega_integrated.shape)

        # Works fast and is correct!
        print('Top Hat filtering to get omega_model_exact')
        self.omega_model_exact  = self.apply_filter(omega_integrated)/(self.Delta_LES/self.filter_width)**3

        # free some memory, field is not needed anymore case
        del self.dirac_xi_fields


    def compute_omega_over_grad_c(self,c_iso):
        '''
        Computes the analytical expression for \omega(c*)/|dc*/dx|
        has to be computed for each c_iso value indivdually
        :return:
        '''

        return (self.m + 1)* c_iso** self.m


    def run_analysis_dirac_FSD(self,filter_width ,filter_type, c_analytical=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        # filter the c and rho field
        print('Filtering c field ...')
        self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

        # # reduce c for computation of conditioned wrinkling factor
        # self.reduce_c(c_min=0.75,c_max=0.85)
        # self.c_filtered_reduced = self.apply_filter(self.c_data_reduced_np)

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)

        # compute omega based on pfitzner
        self.compute_Pfitzner_model()

        # compute omega_model_exact with FSD from Pfitzner
        self.compute_omega_FSD()

        # print('write omega_DNS')
        # filename = join(self.case,
        #                 'filter_width_' + str(self.filter_width) + '_omega_DNS.npy')
        # np.save(filename,self.omega_DNS)

        # creat dask array and reshape all data
        # a bit nasty for list in list as of variable c_iso values
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx**3,1),
                                    self.omega_model_exact.reshape(self.Nx**3,1),
                                    self.omega_DNS_filtered.reshape(self.Nx**3,1)
                                  ])

        filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_FSD_xi_alt.csv')

        self.dataArray_dd = dd.io.from_dask_array(dataArray_da,columns=['c_bar','omega_model_exact','omega_DNS_filtered'])

        # filter the data set and remove unecessary entries
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] > 0.01]
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] < 0.99]


        print('Computing data array ...')
        self.dataArray_df = self.dataArray_dd.sample(frac=0.3).compute()

        print('Writing output to csv ...')
        self.dataArray_df.to_csv(filename,index=False)
        print('Data has been written.\n\n')


##################################################
# implemented 17.3.20

class data_binning_dirac_compare(data_binning_dirac_FSD_alt):
    ''' compares the different Sigma computations
        1. based on slices and convolution with corresponding \omega (FSD assumption)
        2. based on c_iso = 0.85
        3. based on c_iso = c_bar
    '''

    def __init__(self,case, bins, eps_factor,c_iso_values):
        # extend super class with additional input parameters
        super(data_binning_dirac_compare, self).__init__(case, bins, eps_factor,c_iso_values)

        print('This is the version to compare the different Sigmas ...')


    def compute_phi_c(self,c_iso):
        '''
        computes the difference between abs(c(x)-c_iso)
        see Pfitzner notes Eq. 3
        :param c_iso: np vector (1D np.array)
        :return: 1D np.array
        '''

        c_data_vec = self.c_data_np.reshape(self.Nx**3)

        return abs(c_data_vec - c_iso)


    def compute_Xi_iso_dirac_c(self,c_iso):
        '''

        :param c_iso: 3D or 1D np.array
        :return: 3D np.array of the Xi field
        '''

        # make sure c_iso is a vector!
        if np.ndim(c_iso) > 1:
            c_iso = c_iso.reshape(self.Nx**3)

        # check if self.grad_c_DNS was computed, if not -> compute it
        if type(self.grad_c_DNS) is None:
            self.grad_c_DNS = self.compute_DNS_grad_4thO()

        grad_c_DNS_vec = self.grad_c_DNS.reshape(self.Nx**3)

        c_phi = self.compute_phi_c(c_iso)
        dirac = self.compute_dirac_cos(c_phi)
        dirac_times_grad_c_vec = (dirac * grad_c_DNS_vec)
        # print('dirac_imes_grad_c: ', dirac_times_grad_c)

        # convert from vector back to 3D array
        dirac_times_grad_c_arr = dirac_times_grad_c_vec.reshape(self.Nx,self.Nx,self.Nx)

        dirac_LES_filter = self.apply_filter(dirac_times_grad_c_arr) #self.compute_LES_cell_sum(dirac_times_grad_c_arr)

        Xi_iso_filtered = dirac_LES_filter * (self.filter_width**3 / self.filter_width**2)  # Conversion to Xi-space!

        return Xi_iso_filtered


    def run_analysis_dirac_compare(self,filter_width ,filter_type, c_analytical=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        # filter the c and rho field
        print('Filtering c field ...')
        self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)

        # compute self.omega_model_cbar: the Pfitzner model for the planar flame
        self.compute_Pfitzner_model()

        # 1st method
        # compute omega_model_exact with FSD from Pfitzner
        self.compute_omega_FSD()

        # 2nd method
        Xi_iso_085 = self.compute_Xi_iso_dirac_c(c_iso=0.85)

        # 3rd method
        Xi_iso_cbar = self.compute_Xi_iso_dirac_c(c_iso=self.c_filtered)

        # 4th method
        wrinkling_factor = self.get_wrinkling(order='4th')


        # prepare data for output to csv file
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx**3,1),
                                  self.omega_model_exact.reshape(self.Nx**3,1),
                                  self.omega_DNS_filtered.reshape(self.Nx**3,1),
                                  self.omega_model_cbar.reshape(self.Nx**3,1),
                                  Xi_iso_085.reshape(self.Nx**3,1),
                                  Xi_iso_cbar.reshape(self.Nx**3,1),
                                  wrinkling_factor.reshape(self.Nx**3,1)
                                  ])

        filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_compare.csv')

        self.dataArray_dd = dd.io.from_dask_array(dataArray_da,columns=
                                                  ['c_bar',
                                                   'omega_model_FSD',
                                                   'omega_DNS_filtered',
                                                   'omega_model_planar',
                                                   'Xi_iso_085',
                                                   'Xi_iso_cbar',
                                                   'wrinkling_factor'
                                                   ])

        # filter the data set and remove unecessary entries
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] > 0.01]
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] < 0.99]

        print('Computing data array ...')
        self.dataArray_df = self.dataArray_dd.sample(frac=0.3).compute()

        print('Writing output to csv ...')
        self.dataArray_df.to_csv(filename,index=False)
        print('Data has been written.\n\n')

        