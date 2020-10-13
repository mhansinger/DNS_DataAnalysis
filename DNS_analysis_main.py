'''
This is to read in the binary data File for the high pressure bunsen data and planar turbulent flame

use: dns_analysis_UVW -> includes the Velocity data for the post processing

@author: mhansinger

last change: April 2020
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import dask.dataframe as dd
from numba import jit, cuda
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

#external cython functions
from external_cython.compute_uprime_cython import compute_U_prime_cython
from external_cython.compute_gradU_LES_4thO_cython import compute_gradU_LES_4thO_cython
from external_cython.compute_DNS_grad_4thO_cython import compute_DNS_grad_4thO_cython
from external_cython.compute_gradU_LES_4thO_tensor_cython import compute_gradU_LES_4thO_tensor_cython
from external_cython.compute_LES_grad_4thO_tensor_cython import compute_LES_grad_4thO_tensor_cython


class dns_analysis_base(object):
    # Base class. To be inherited from

    def __init__(self, case):
        '''
        # CONSTRUCTOR
        :param case:    case name: NX512, 1bar, 5bar or 10bar
        '''

        # THIS NAME IS GIVEN BY THE OUTPUT FROM FORTRAN CODE, COULD CHANGE...
        self.c_by_rho_path = join(case,'rho_by_c.dat')
        self.rho_path = join(case,'rho.dat')
        self.c_path = join(case,'c.dat')

        self.c_data_np = None
        self.data_rho = None
        self.case = case

        # Filter width of the LES cell: is filled later
        self.filter_width = None

        self.every_nth = None

        # gradient of c on the DNS mesh
        self.grad_c_DNS = None

        # gradient of c on the LES mesh
        self.grad_c_LES = None

        if self.case is '1bar':
            # NUMBER OF DNS CELLS IN X,Y,Z DIRECTION
            self.Nx = self.Ny = self.Nz = 250
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
            self.Nx = self.Ny = self.Nz = 560
            self.bfact = 7128.3
            self.Re = 892 # 500
            self.delta_x = 1/432
            self.p = 5
            m = 4.4545
            beta=6
            alpha=0.81818
        elif self.case=='10bar':
            self.Nx = self.Ny = self.Nz = 795
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
            self.Nx = self.Ny = self.Nz = 150
            self.bfact = 7364.0
            self.Re = 100
            self.delta_x = 1/188
            self.p = 1
        elif self.case.startswith('NX512'):
            # check: Parameter_PlanarFlame.xlsx
            self.Nx = self.Ny = self.Nz = 512
            self.bfact = 3675
            self.Re = 50
            self.delta_x = 1/220    # Klein nochmal fragen! -> 220 stimmt!
            self.p = 1
            m = 4.4545
            beta=6
            alpha=0.81818
        elif self.case is 'planar_flame_test':
            # check: Parameter_PlanarFlame.xlsx
            print('\n################\nThis is the laminar planar test case!\n################\n')
            self.Nx = self.Ny = self.Nz = 512
            self.bfact = 3675
            self.Re = 50
            self.delta_x = 1/220
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

        # normalizing pressure
        self.p_0 = 1

        # Variables for FILTERING
        self.c_filtered = np.zeros((self.Nx,self.Ny,self.Nz))
        #self.rho_filtered = np.zeros((self.Nx,self.Ny,self.Nz))
        #self.c_filtered_clipped = np.zeros((self.Nx,self.Ny,self.Nz))       # c für wrinkling factor nur zw 0.75 und 0.85

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
        '''
        reads in the c field and stores it in self.c_data_np as np.array. 3D!!
        :return: nothing
        '''

        print('Reading in c, rho*c, rho data...')

        try:
            c_data_vec = pd.read_csv(self.c_path,names=['c']).values.astype(dtype=np.float32)
            self.c_data_np = c_data_vec.reshape(self.Nx,self.Ny,self.Nz)

            crho_data_vec = pd.read_csv(self.c_by_rho_path,names=['rho_c']).values.astype(dtype=np.float32)
            self.rho_c_data_np = crho_data_vec.reshape(self.Nx,self.Ny,self.Nz)

            rho_data_vec = pd.read_csv(self.rho_path,names = ['rho']).values.astype(dtype=np.float32)
            self.rho_data_np = rho_data_vec.reshape(self.Nx,self.Ny,self.Nz)

        except OSError:
            print('c.dat not found, compute it from rho_c.dat and rho.dat')

            try:
                self.data_rho_c = dd.read_csv(self.c_by_rho_path,names=['rho_c'])
            except:
                sys.exit('No data for C_rho')

            try:
                self.data_rho = dd.read_csv(self.rho_path,names = ['rho'])
            except:
                sys.exit('No data for rho')

            # transform the data into an array and reshape it to 3D
            self.rho_data_np = self.data_rho.to_dask_array(lengths=True).reshape(self.Nx,self.Ny,self.Nz).compute()
            self.rho_c_data_np = self.data_rho_c.to_dask_array(lengths=True).reshape(self.Nx,self.Ny,self.Nz).compute()

            # progress variable
            self.c_data_np = self.rho_c_data_np / self.rho_data_np
            #self.c_data_reduced_np = self.rho_c_data_np / self.rho_data_np      # reduce c between 0.75 and 0.85


    def apply_filter(self,data):
        '''
        method to filter the data array. Input needs to be 3D
        Either Gauss or TopHat filter
        :param data: 3D data np.array which is to filter
        :return:
        '''

        # check if data is 3D array
        try:
            assert type(data) == np.ndarray
        except AssertionError:
            print('Only np.ndarrays are allowed in Gauss_filter!')

        if len(data.shape) == 1:
            data = data.reshape(self.Nx,self.Ny,self.Nz)


        if self.filter_type == 'GAUSS':
            self.sigma_xyz = [int(self.filter_width/2), int(self.filter_width/2) ,int(self.filter_width/2)]
            data_filtered = sp.ndimage.filters.gaussian_filter(data, self.sigma_xyz, truncate=1.0, mode='wrap')
            return data_filtered

        elif self.filter_type == 'TOPHAT':
            data_filtered = sp.ndimage.filters.uniform_filter(data, [self.filter_width,self.filter_width,self.filter_width],mode='wrap')
            return data_filtered

        elif self.filter_type == 'TOPHAT_REFLECT':
            data_filtered = sp.ndimage.filters.uniform_filter(data, [self.filter_width,self.filter_width,self.filter_width],mode='reflect')
            return data_filtered

        else:
            sys.exit('No fitler type provided ...')


    def set_gaussian_kernel(self):
        '''
        Set the gaussian Kernel. Probably not used...
        :return:
        '''
        size = int(self.filter_width)
        vector = np.linspace(-self.filter_width,self.filter_width,2*self.filter_width+1)
        x,y,z = np.meshgrid(vector, vector, vector)
        x = x * self.delta_x
        y = y * self.delta_x
        z = z * self.delta_x

        self.gauss_kernel = np.sqrt(12)/self.Delta_LES/np.sqrt(2*np.pi) * \
                            np.exp(-6*(x**2/self.Delta_LES**2 +y**2/self.Delta_LES**2 + z**2/self.Delta_LES**2))


    def get_wrinkling(self,order='2nd'):
        '''
        computes the wrinkling factor
        :param order: 2nd or 4th order
        :return: wrinkling factor for the whole field
        '''

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
        '''
        Compute the magnitude of the gradient of the DNS c-field, based on neighbour cells
        2nd Order central differencing
        :return: gradient of c for DNS: |\nabla grad(c)|
        '''

        print('Computing DNS gradients...')

        # create empty array
        grad_c_DNS = np.zeros([self.Nx,self.Ny,self.Nz])

        # compute gradients from the boundaries away ...
        for l in range(2,self.Nx-2):
            for m in range(2,self.Ny-2):
                for n in range(2,self.Nz-2):
                    this_DNS_gradX = (self.c_data_np[l+1, m, n] - self.c_data_np[l-1,m, n])/(2 * self.delta_x)
                    this_DNS_gradY = (self.c_data_np[l, m+1, n] - self.c_data_np[l, m-1, n]) / (2 * self.delta_x)
                    this_DNS_gradZ = (self.c_data_np[l, m, n+1]- self.c_data_np[l, m, n-1]) / (2 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_DNS_magGrad_c = np.sqrt(this_DNS_gradX**2 + this_DNS_gradY**2 + this_DNS_gradZ**2)

                    grad_c_DNS[l,m,n] = this_DNS_magGrad_c

        return grad_c_DNS

    def compute_DNS_grad_4thO(self):
        '''
        Compute the magnitude of the gradient of the DNS c-field, based on neighbour cells
        4th Order central differencing
        :return: gradient of c for DNS: |\nabla grad(c)|
        '''

        print('Computing DNS gradients 4th Order...')

        # create empty array
        grad_c_DNS = np.zeros([self.Nx,self.Ny,self.Nz])

        # compute gradients from the boundaries away ...
        for l in range(2,self.Nx-2):
            for m in range(2,self.Ny-2):
                for n in range(2,self.Nz-2):
                    this_DNS_gradX = (-self.c_data_np[l+2, m, n] + 8*self.c_data_np[l+1,m, n] - 8*self.c_data_np[l-1,m, n] + self.c_data_np[l-2, m, n])/(12 * self.delta_x)
                    this_DNS_gradY = (-self.c_data_np[l, m+2, n] + 8*self.c_data_np[l,m+1, n] - 8*self.c_data_np[l,m-1, n] + self.c_data_np[l, m-2, n])/(12 * self.delta_x)
                    this_DNS_gradZ = (-self.c_data_np[l, m, n+2] + 8*self.c_data_np[l,m, n+1] - 8*self.c_data_np[l,m, n-1] + self.c_data_np[l, m, n+2])/(12 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_DNS_magGrad_c = np.sqrt(this_DNS_gradX**2 + this_DNS_gradY**2 + this_DNS_gradZ**2)

                    grad_c_DNS[l,m,n] = this_DNS_magGrad_c

        return grad_c_DNS

    #@jit(nopython=True, parallel=True)
    def compute_DNS_grad_reduced(self):
        # # computes the flame surface area in the DNS based on gradients of c of neighbour cells
        # # for the reduced c
        # #width = 1
        #
        # print('Computing DNS gradients for c reduced...')
        #
        # # create empty array
        # grad_c_DNS = np.zeros([self.Nx,self.Ny,self.Nz])
        #
        # # compute gradients from the boundaries away ...
        # for l in range(1,self.Nx-1):
        #     for m in range(1,self.Nx-1):
        #         for n in range(1,self.Nx-1):
        #             this_DNS_gradX = (self.c_data_reduced_np[l+1, m, n] - self.c_data_reduced_np[l-1,m, n])/(2 * self.delta_x)
        #             this_DNS_gradY = (self.c_data_reduced_np[l, m+1, n] - self.c_data_reduced_np[l, m-1, n]) / (2 * self.delta_x)
        #             this_DNS_gradZ = (self.c_data_reduced_np[l, m, n+1] - self.c_data_reduced_np[l, m, n-1]) / (2 * self.delta_x)
        #             # compute the magnitude of the gradient
        #             this_DNS_magGrad_c = np.sqrt(this_DNS_gradX**2 + this_DNS_gradY**2 + this_DNS_gradZ**2)
        #
        #             grad_c_DNS[l,m,n] = this_DNS_magGrad_c
        #
        # return grad_c_DNS
        return NotImplementedError

    #@jit(nopython=True)
    def compute_LES_grad(self):
        '''
        Compute the magnitude of the gradient of the filtered (LES) c-field, based on neighbour cells on the DNS mesh
        2nd Order central differencing
        :return: gradient of c for LES: |\nabla grad(c)|
        '''

        print('Computing LES gradients on DNS mesh ...')

        # create empty array
        self.grad_c_LES = np.zeros([self.Nx, self.Ny, self.Nz])

        # compute gradients from the boundaries away ...
        for l in range(2,self.Nx-2):
            for m in range(2,self.Ny-2):
                for n in range(2,self.Nz-2):
                    this_LES_gradX = (self.c_filtered[l + 1, m, n] - self.c_filtered[l - 1, m, n]) / (2 * self.delta_x)
                    this_LES_gradY = (self.c_filtered[l, m + 1, n] - self.c_filtered[l, m - 1, n]) / (2 * self.delta_x)
                    this_LES_gradZ = (self.c_filtered[l, m, n + 1] - self.c_filtered[l, m, n - 1]) / (2 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)

                    self.grad_c_LES[l, m, n] = this_LES_magGrad_c

        return self.grad_c_LES

    def compute_LES_grad_4thO(self):
        '''
        Compute the magnitude of the gradient of the filtered (LES) c-field, based on neighbour cells on the DNS mesh
        4th Order central differencing
        :return: gradient of c for LES: |\nabla grad(c)|
        '''

        print('Computing LES 4th order gradients on DNS mesh ...')

        # create empty array
        self.grad_c_LES = np.zeros([self.Nx, self.Ny, self.Nz])

        # compute gradients from the boundaries away ...
        for l in range(2, self.Nx - 2):
            for m in range(2, self.Ny - 2):
                for n in range(2, self.Nz - 2):
                    this_LES_gradX = (-self.c_filtered[l + 2, m, n] + 8*self.c_filtered[l + 1, m, n] - 8*self.c_filtered[l - 1, m, n] + self.c_filtered[l - 2, m, n]) / (12 * self.delta_x)
                    this_LES_gradY = (-self.c_filtered[l, m + 2, n] + 8*self.c_filtered[l, m+1, n] - 8*self.c_filtered[l, m-1, n] + self.c_filtered[l, m-2, n]) / (12 * self.delta_x)
                    this_LES_gradZ = (-self.c_filtered[l, m, n+2] + 8*self.c_filtered[l, m, n+1] - 8*self.c_filtered[l, m, n-1] + self.c_filtered[l, m, n-2]) / (12 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)

                    self.grad_c_LES[l, m, n] = this_LES_magGrad_c

        return self.grad_c_LES

    #@jit(nopython=True)
    def compute_LES_grad_reduced(self):
        # # computes the flame surface area in the DNS based on gradients of c of neighbour DNS cells
        #
        # print('Computing LES gradients on DNS mesh for c reduced ...')
        #
        # # create empty array
        # self.grad_c_LES = np.zeros([self.Nx, self.Ny, self.Nz])
        #
        # # compute gradients from the boundaries away ...
        # for l in range(1, self.Nx - 1):
        #     for m in range(1, self.Nx - 1):
        #         for n in range(1, self.Nx - 1):
        #             this_LES_gradX = (self.c_filtered_reduced[l + 1, m, n] - self.c_filtered_reduced[l - 1, m, n]) / (2 * self.delta_x)
        #             this_LES_gradY = (self.c_filtered_reduced[l, m + 1, n] - self.c_filtered_reduced[l, m - 1, n]) / (2 * self.delta_x)
        #             this_LES_gradZ = (self.c_filtered_reduced[l, m, n + 1] - self.c_filtered_reduced[l, m, n - 1]) / (2 * self.delta_x)
        #             # compute the magnitude of the gradient
        #             this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)
        #
        #             self.grad_c_LES[l, m, n] = this_LES_magGrad_c
        #
        # return self.grad_c_LES
        return NotImplementedError

    #@jit(nopython=True)
    def compute_LES_grad_onLES(self):
        '''
        Compute the magnitude of the gradient of the filtered (LES) c-field, based on neighbour cells on the LES mesh
        2nd Order central differencing
        :return: gradient of c for LES: |\nabla grad(c)|
        '''

        print('Computing LES gradients on LES mesh ...')

        # create empty array
        self.grad_c_LES = np.zeros([self.Nx, self.Ny, self.Nz])

        # compute gradients from the boundaries away ...
        for l in range(self.filter_width, self.Nx - self.filter_width):
            for m in range(self.filter_width, self.Ny - self.filter_width):
                for n in range(self.filter_width, self.Nz - self.filter_width):
                    this_LES_gradX = (self.c_filtered[l + self.filter_width, m, n] - self.c_filtered[l - self.filter_width, m, n]) / (2 * self.delta_x * self.filter_width)
                    this_LES_gradY = (self.c_filtered[l, m + self.filter_width, n] - self.c_filtered[l, m - self.filter_width, n]) / (2 * self.delta_x * self.filter_width)
                    this_LES_gradZ = (self.c_filtered[l, m, n + self.filter_width] - self.c_filtered[l, m, n - self.filter_width]) / (2 * self.delta_x * self.filter_width)
                    # compute the magnitude of the gradient
                    this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)

                    self.grad_c_LES[l, m, n] = this_LES_magGrad_c

        return self.grad_c_LES

    #@jit(nopython=True)
    def compute_LES_grad_onLES_reduced(self):
        # # computes the flame surface area in the DNS based on gradients of c of neighbour LES cells
        #
        # print('Computing LES gradients on LES mesh ...')
        #
        # # create empty array
        # self.grad_c_LES = np.zeros([self.Nx, self.Ny, self.Nz])
        #
        # # compute gradients from the boundaries away ...
        # for l in range(self.filter_width, self.Nx - self.filter_width):
        #     for m in range(self.filter_width, self.Nx - self.filter_width):
        #         for n in range(self.filter_width, self.Nx - self.filter_width):
        #             this_LES_gradX = (self.c_filtered_reduced[l + self.filter_width, m, n] - self.c_filtered_reduced[l - self.filter_width, m, n]) / (2 * self.delta_x * self.filter_width)
        #             this_LES_gradY = (self.c_filtered_reduced[l, m + self.filter_width, n] - self.c_filtered_reduced[l, m - self.filter_width, n]) / (2 * self.delta_x * self.filter_width)
        #             this_LES_gradZ = (self.c_filtered_reduced[l, m, n + self.filter_width] - self.c_filtered_reduced[l, m, n - self.filter_width]) / (2 * self.delta_x * self.filter_width)
        #             # compute the magnitude of the gradient
        #             this_LES_magGrad_c = np.sqrt(this_LES_gradX ** 2 + this_LES_gradY ** 2 + this_LES_gradZ ** 2)
        #
        #             self.grad_c_LES[l, m, n] = this_LES_magGrad_c
        #
        # return self.grad_c_LES
        return NotImplementedError


    def compute_isoArea(self,c_iso):
        '''
        Computes the flame surface iso Area in a LES cell using the MARCHING CUBES ALGORITHM
        :param c_iso: define a c_iso value to compute the surface
        :return: isoArea_coefficient = A_turbulent/A_planar
        '''
        print('Computing the surface for c_iso: ', c_iso)
        # print('Currently in timing test mode!')

        half_filter = int(self.filter_width/2)

        # reference area of planar flame
        A_planar = (self.filter_width - 1)**2

        iterpoints = self.Nx * self.Ny *self.Nz
        # progress bar
        bar = ChargingBar('Processing', max=iterpoints)

        isoArea_coefficient = np.zeros((self.Nx,self.Ny,self.Nz))

        for l in range(half_filter, self.Nx - half_filter, self.every_nth):
            for m in range(half_filter, self.Ny - half_filter, self.every_nth):
                for n in range(half_filter, self.Nz - half_filter, self.every_nth):

                    this_LES_box = (self.c_data_np[l-half_filter : l+half_filter,
                                                  m-half_filter : m+half_filter,
                                                  n-half_filter : n+half_filter])

                    # this works only if the c_iso value is contained in my array
                    # -> check if array contains values above AND below iso value
                    if np.any(np.where(this_LES_box < c_iso)) and np.any(np.any(np.where(this_LES_box > c_iso))):

                        verts, faces = measure.marching_cubes_classic(this_LES_box, c_iso)
                        iso_area = measure.mesh_surface_area(verts=verts, faces=faces)

                    else:
                        iso_area = 0

                    isoArea_coefficient[l, m, n] = iso_area / A_planar

                    # iterbar
                    bar.next()

        bar.finish()

        return isoArea_coefficient

    def compute_isoArea_dynamic(self):
        '''
        Computes the flame surface iso Area in a LES cell using the MARCHING CUBES ALGORITHM
        based on c_bar in the respective LES cell -> dynamic definition of c_iso
        :return: isoArea_coefficient = A_turbulent/A_planar
        '''
        print('Computing the surface for c_iso based on c_bar ')
        # print('Currently in timing test mode!')

        half_filter = int(self.filter_width/2)

        # reference area of planar flame
        A_planar = (self.filter_width - 1)**2

        iterpoints = (self.Nx)**3
        # progress bar
        bar = ChargingBar('Processing', max=iterpoints)

        isoArea_coefficient = np.zeros((self.Nx,self.Ny,self.Nz))

        for l in range(half_filter, self.Nx - half_filter, self.every_nth):
            for m in range(half_filter, self.Ny - half_filter, self.every_nth):
                for n in range(half_filter, self.Nz - half_filter, self.every_nth):

                    this_LES_box = (self.c_data_np[l-half_filter : l+half_filter,
                                                  m-half_filter : m+half_filter,
                                                  n-half_filter : n+half_filter])

                    # compute c_bar of current LES box
                    this_c_bar = np.mean(this_LES_box)[0]
                    c_iso = this_c_bar
                    print('c_iso: %f' % c_iso)
                    # this works only if the c_iso value is contained in my array
                    # -> check if array contains values above AND below iso value
                    if np.any(np.where(this_LES_box < c_iso)) and np.any(np.any(np.where(this_LES_box > c_iso))):

                        verts, faces = measure.marching_cubes_classic(this_LES_box, c_iso)
                        iso_area = measure.mesh_surface_area(verts=verts, faces=faces)

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

        isoArea_coefficient = np.zeros((self.Nx,self.Ny,self.Nz))

        isoArea_list = Parallel(n_jobs=4)(delayed(self.compute_this_LES_box)(l,m,n, half_filter,c_iso,isoArea_coefficient)
                           for n in DNS_range
                           for m in DNS_range
                           for l in DNS_range)

        # reshape isoArea_list into 3D np.array
        isoArea_coefficient = np.array(isoArea_list).reshape(self.Nx,self.Ny,self.Nz)

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
        '''
        Filters the DNS gradients
        2nd Order
        :return: grad_DNS_filtered
        '''

        # compute dask delayed object
        grad_c_DNS = self.compute_DNS_grad()

        grad_DNS_filtered = self.apply_filter(grad_c_DNS)

        return grad_DNS_filtered

    def compute_filter_DNS_grad_4thO(self):
        '''
        Filters the DNS gradients
        4th Order
        :return: grad_DNS_filtered
        '''

        # compute dask delayed object
        grad_c_DNS = self.compute_DNS_grad_4thO()

        grad_DNS_filtered = self.apply_filter(grad_c_DNS)

        return grad_DNS_filtered


    def compute_filter_DNS_grad_reduced(self):
    # # compute filtered DNS reaction rate
    #
    #     # compute dask delayed object
    #     grad_c_DNS = self.compute_DNS_grad_reduced()
    #
    #     grad_DNS_filtered = self.apply_filter(grad_c_DNS)
    #
    #     return grad_DNS_filtered
        return NotImplementedError


    def compute_RR_DNS(self):
        '''
        Computes the reaction rate of the DNS data.
        See Pfitzner, FTC, 2019, Eq. 4, with Lambda = 18.97
        :return: omega_DNS
        '''

        Lambda = 18.97

        c_data_np_vector = self.c_data_np.reshape(self.Nx*self.Ny*self.Nz)

        # according to Pfitzner implementation
        exponent = - self.beta*(1 - c_data_np_vector) / (1 - self.alpha*(1 - c_data_np_vector))

        this_RR_reshape_DNS_Pfitz  = Lambda * ((1 - self.alpha * (1 - c_data_np_vector))) ** (-1) \
                                     * (1 - c_data_np_vector) * np.exp(exponent)

        # reshape it to 3D array
        RR_DNS = this_RR_reshape_DNS_Pfitz.reshape(self.Nx,self.Ny,self.Nz)

        return RR_DNS


    def filter_RR_DNS(self):
        '''
        Filters the DNS reaction rates
        :return: omega_DNS_filtered
        '''

        RR_DNS_filtered = self.apply_filter(self.omega_DNS)

        return RR_DNS_filtered


    def compute_RR_LES(self):
        '''
        Computes the filtered reaction rates based on the filtered c field (c_bar)
        See Pfitzner, FTC, 2019, Eq. 4, with Lambda = 18.97
        :return: omega_LES
        '''

        Lambda = 18.97

        # according to Pfitzner implementation
        exponent = - self.beta*(1-self.c_filtered.reshape(self.Nx*self.Ny*self.Nz)) / (1 - self.alpha*(1 - self.c_filtered.reshape(self.Nx*self.Ny*self.Nz)))
        #this_RR_reshape_DNS = self.bfact*self.rho_data_np.reshape(self.Nx*self.Ny*self.Nz)*(1-self.c_data_np.reshape(self.Nx*self.Ny*self.Nz))*np.exp(exponent)

        this_RR_reshape_LES_Pfitz  = Lambda * ((1 - self.alpha * (1 - self.c_data_np.reshape(self.Nx*self.Ny*self.Nz)))) ** (-1) \
                                     * (1 - self.c_data_np.reshape(self.Nx*self.Ny*self.Nz)) * np.exp(exponent)

        # reshape it to 3D array
        RR_LES = this_RR_reshape_LES_Pfitz.reshape(self.Nx,self.Ny,self.Nz)

        return RR_LES


    # added Nov. 2018: Implementation of Pfitzner's analytical boundaries
    # getter and setter for c_Mean as a protected

    def compute_flamethickness(self):
        '''
        Copmutes the flame thicknes
        See Pfitzner, FTC, 2019, Eq. 17 according to analytical model (not numerical!)
        :param m:
        :return: flame thicknes dth
        '''

        return (self.m + 1) ** (1 / self.m + 1) / self.m

    # compute self.delta_x_0
    def compute_delta_0(self,c):
        '''
        See Pfitzner, FTC, 2019, Eq. 38
        :param c: usually c_0
        :param m: steepness of the flame front; not sure how computed
        :return: computes self.delta_x_0, needed for c_plus and c_minus
        '''
        return (1 - c ** self.m) / (1-c)

    def compute_c_m(self,xi):
        '''
        computes c_m
        See Pfitzner, FTC, 2019, Eq. 12
        :param xi: transformed coordinate
        :return: c_m
        '''
        return (1 + np.exp(- self.m * xi)) ** (-1 / self.m)

    def compute_xi_m(self,c):
        '''
        computes xi_m
        See Pfitzner, FTC, 2019, Eq. 13
        :param c: c progress variable
        :return: xi_m
        '''
        return 1 / self.m * np.log(c ** self.m / (1 - c ** self.m))

    def compute_s(self,c):
        '''
        See Pfitzner, FTC, 2019, Eq. 39
        :param c: progress variable
        :param Delta_LES: Delta_LES (scaled filter width)
        :return: s
        '''
        s = np.exp( - self.Delta_LES / 7) * ((np.exp(self.Delta_LES / 7) - 1) * np.exp(2 * (c - 1) * self.m) + c)
        return s

    # compute the values for c_minus
    def compute_c_minus(self):
        '''
        compute c_minus field and update self.c_minus array
        See Pfitzner, FTC, 2019, Eq. 40
        :return: nothing
        '''

        # self.c_filtered.reshape(self.Nx*self.Ny*self.Nz) = c_bar in der ganzen domain als vector
        this_s = self.compute_s(self.c_filtered.reshape(self.Nx*self.Ny*self.Nz))
        this_delta_0 = self.compute_delta_0(this_s)

        self.c_minus = (np.exp(self.c_filtered.reshape(self.Nx*self.Ny*self.Nz)* this_delta_0 * self.Delta_LES) - 1) / \
                       (np.exp(this_delta_0 * self.Delta_LES) - 1)


    # Analytical c_minus (Eq. 35)
    def compute_c_minus_analytical(self):
        '''
        Analytical approximation to obtain c_minus. Updates self.c_minus and self.c_plus array
        See Pfitzner, FTC,2019, Eq. 35
        :return: nothing
        '''
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
        self.c_minus = f_c_minus(self.c_filtered.reshape(self.Nx*self.Ny*self.Nz))
        self.c_plus = f_c_plus(self.c_filtered.reshape(self.Nx*self.Ny*self.Nz))


    def I_1(self,c):
        '''
        See Pfitzner, FTC, 2019, Eq. 35
        :param c:
        :return: Hypergeometric function
        '''
        return c / self.Delta_LES * special.hyp2f1(1, 1 / self.m, 1 + 1 / self.m, c ** self.m)

    def compute_c_plus(self):
        '''
        Update self.c_plus field
        See Pfitzner, FTC, 2019, Eq. 13
        :param c: c_minus
        :return: nothing
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

        # reshape to 3D array
        return omega_cbar.reshape(self.Nx,self.Ny,self.Nz)


    def model_omega(self,c):
        '''
        Analytical model for flat flame source term omega_model
        See Pfitzner, FTC, Eq. 14
        :param c:
        :return: omega_model planar
        '''

        return (self.m + 1) * (1 - c ** self.m) * c ** (self.m + 1)


    def analytical_omega(self, c):
        '''
        Analytical omega source term based on c value
        Similar to method 'compute_RR_DNS' but does not apply to whole c field
        See Pfitzner, FTC, Eq. 4
        :param c: progress variable
        :return: computes the analytical omega for given c_bar!
        '''

        Lambda = 18.97
        print('Computing omega DNS ...')

        exponent = - (self.beta * (1 - c)) / (1 - self.alpha * (1 - c))

        # om_Klein = self.bfact*self.rho_bar*(1-c)*np.exp(exponent)
        om_Pfitzner = Lambda * ((1 - self.alpha * (1 - c))) ** (-1) * (1 - c) * np.exp(exponent)

        return om_Pfitzner


    def compute_Pfitzner_model(self):
        '''
        Sequential computation to obtain omega_model
        See Pfitzner, FTC, 2019 for more details
        :param self.omega_model_cbar: is the modeled omega from the laminar planar flame
        :param self.omega_DNS: is the (real) omega field from the DNS data
        :param self.omega_DNS_filtered: is the filtered omega field from DNS data; this is the bench mark to compare with
        :return nothing
        '''

        # switch between the computation modes
        if self.c_analytical is True:       #--> seems to be better
            self.compute_c_minus_analytical()
        else:
            self.compute_c_minus()
            self.compute_c_plus()

        self.omega_model_cbar = self.compute_model_omega_bar()

        #self.omega_model_cbar = self.model_omega(self.c_filtered.reshape(self.Nx*self.Ny*self.Nz))
        self.omega_DNS = self.analytical_omega(self.c_data_np.reshape(self.Nx*self.Ny*self.Nz))

        if len(self.omega_DNS.shape) == 1:
            self.omega_DNS = self.omega_DNS.reshape(self.Nx,self.Ny,self.Nz)

        # filter the DNS reaction rate
        print('Filtering omega DNS ...')

        self.omega_DNS_filtered = self.apply_filter(self.omega_DNS)


#####################################################
class dns_analysis_wrinkling(dns_analysis_base):

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
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x * self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
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

            pd.DataFrame(data=np.hstack([
                                self.omega_DNS.reshape(self.Nx*self.Ny*self.Nz,1),
                                self.omega_DNS_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                om_iso.reshape(self.Nx*self.Ny*self.Nz,1),
                                om_wrinkl.reshape(self.Nx*self.Ny*self.Nz,1),
                                self.c_filtered.reshape(self.Nx*self.Ny*self.Nz, 1)]),
                                columns=['omega_DNS',
                                        'omega_filtered',
                                        'omega_model_by_isoArea',
                                        'omega_model_by_wrinkling',
                                        'c_bar']).to_csv(filename)

        # creat dask array and reshape all data
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                   self.wrinkling_factor.reshape(self.Nx*self.Ny*self.Nz,1),
                                   isoArea_coefficient.reshape(self.Nx*self.Ny*self.Nz,1),
                                   self.omega_model_cbar.reshape(self.Nx*self.Ny*self.Nz,1),
                                   self.omega_DNS_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                   self.c_plus.reshape(self.Nx*self.Ny*self.Nz,1),
                                   self.c_minus.reshape(self.Nx*self.Ny*self.Nz,1)])


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
                                                      'omega_model',
                                                      'omega_DNS_filtered',
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


#######################################################
# NEW CLASS FOR THE CLUSTER
#TODO: Evtl wieder löschen...

class dns_analysis_cluster(dns_analysis_base):

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
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

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

            pd.DataFrame(data=np.hstack([self.omega_DNS.reshape(self.Nx*self.Ny*self.Nz,1),
                               self.omega_DNS_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                               om_iso.reshape(self.Nx*self.Ny*self.Nz,1),
                               om_wrinkl.reshape(self.Nx*self.Ny*self.Nz,1),
                               self.c_filtered.reshape(self.Nx*self.Ny*self.Nz, 1)]),
                               columns=['omega_DNS',
                                        'omega_filtered',
                                        'omega_model_by_isoArea',
                                        'omega_model_by_wrinkling',
                                        'c_bar']).to_csv(filename)


        # creat dask array and reshape all data
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                   self.wrinkling_factor.reshape(self.Nx*self.Ny*self.Nz,1),
                                   isoArea_coefficient.reshape(self.Nx*self.Ny*self.Nz,1),
                                   # self.wrinkling_factor_LES.reshape(self.Nx*self.Ny*self.Nz, 1),
                                   # self.wrinkling_factor_reduced.reshape(self.Nx*self.Ny*self.Nz, 1),
                                   # self.wrinkling_factor_LES_reduced.reshape(self.Nx*self.Ny*self.Nz, 1),
                                   self.omega_model_cbar.reshape(self.Nx*self.Ny*self.Nz,1),
                                   self.omega_DNS_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                   #self.omega_LES_noModel.reshape(self.Nx*self.Ny*self.Nz,1),
                                   self.c_plus.reshape(self.Nx*self.Ny*self.Nz,1),
                                   self.c_minus.reshape(self.Nx*self.Ny*self.Nz,1)])


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
# NEW CLASS WITH PFITZNERS SCALING FACTOR

class dns_analysis_dirac(dns_analysis_base):
    # new implementation with numerical delta dirac function

    def __init__(self, case, eps_factor=100, c_iso_values=[0.5]):
        # extend super class with additional input parameters
        super(dns_analysis_dirac, self).__init__(case)
        self.c_iso_values = c_iso_values
        self.eps_factor = eps_factor
        # the convolution of dirac x grad_c at all relevant iso values and LES filtered will be stored here
        self.Xi_iso_filtered = np.zeros((self.Nx,self.Ny,self.Nz,len(self.c_iso_values)))

        print('You are using the Dirac version...')

    def compute_phi_c(self,c_iso):
        '''
        computes the difference between abs(c(x)-c_iso)
        :param c_iso:
        :return: self.c_data_np - c_iso
        '''
        return abs(self.c_data_np - c_iso)

    def compute_dirac_cos(self,c_phi):
        '''
        :param c_phi: c(x) - c_iso
        :return: numerical dirac delta function
        '''
        eps = self.eps_factor*self.delta_x

        X = c_phi/eps

        # transform to vector
        X_vec = X.reshape(self.Nx*self.Ny*self.Nz)

        dirac_vec = np.zeros(len(X_vec))

        # Fallunterscheidung für X < 0
        for id, x in enumerate(X_vec):
            if x < 1:
                dirac_vec[id] =1/(2*eps) * (1 + np.cos(np.pi * x)) * self.delta_x
                #print('dirac_vec: ',dirac_vec[id])

        # reshape to 3D array
        dirac_array = dirac_vec.reshape(self.Nx,self.Ny,self.Nz)

        return dirac_array


    def compute_Xi_iso_dirac(self):

        # check if self.grad_c_DNS was computed, if not -> compute it
        if self.grad_c_DNS is None:
            self.grad_c_DNS=self.compute_DNS_grad_4thO()

        # loop over the different c_iso values
        for id, c_iso in enumerate(self.c_iso_values):

            c_phi = self.compute_phi_c(c_iso)
            dirac = self.compute_dirac_cos(c_phi)
            self.dirac_times_grad_c = (dirac * self.grad_c_DNS).reshape(self.Nx,self.Ny,self.Nz)
            print('dirac_imes_grad_c: ', self.dirac_times_grad_c)

            # check if integral is 1 for arbitrary
            # Line integral
            print('line integrals over dirac_times_grad(c):')
            #print(np.trapz(dirac_times_grad_c[:, 250, 250]))
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
                self.grad_c_05 = self.grad_c_DNS.reshape(self.Nx,self.Ny,self.Nz)
                self.dirac_05 = dirac.reshape(self.Nx,self.Ny,self.Nz)

            # TODO: check if that is correct!
            dirac_LES_sums = self.compute_LES_cell_sum(self.dirac_times_grad_c)
            self.Xi_iso_filtered[:, :, :, id] = dirac_LES_sums / self.filter_width**2

            #apply TOPHAT filter to dirac_times_grad_c --> Obtain surface
            # Check Eq. 6 from Pfitzner notes (Generalized wrinkling factor)
            # print('Filtering for Xi_iso at c_iso: %f' % c_iso)

            # # stimmt vermutlich nicht...
            # self.Xi_iso_filtered[:,:,:,id] = self.apply_filter(self.dirac_times_grad_c) / self.Delta_LES**2
            # # TTODO: check if self.Delta_LES is correct model parameter here

    def compute_LES_cell_sum(self,input_array):
        # get the sum of values inside an LES cell
        print('computing cell sum')

        try:
            assert len(input_array.shape) == 3
        except AssertionError:
            print('input array must be 3D!')

        output_array = copy.copy(input_array)

        # print('\noutput_array.shape: ',output_array.shape)

        output_array *= 0.0                 # set output array to zero

        half_filter = int(self.filter_width / 2)

        for l in range(half_filter, self.Nx - half_filter, 1):
            for m in range(half_filter, self.Ny - half_filter, 1):
                for n in range(half_filter, self.Nz - half_filter, 1):

                    this_LES_box = (input_array[l - half_filter: l + half_filter,
                                    m - half_filter: m + half_filter,
                                    n - half_filter: n + half_filter])

                    # compute c_bar of current LES box
                    output_array[l,m,n] = this_LES_box.sum()

        return output_array

    def run_analysis_dirac(self,filter_width ,filter_type, c_analytical=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :param Parallel: use False
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
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

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
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.Xi_iso_filtered[:,:,:,0].reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.omega_model_cbar.reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.omega_DNS_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.grad_c_05.reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.dirac_05.reshape(self.Nx*self.Ny*self.Nz, 1)
                                  ])

        filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_dirac.csv')

        self.dataArray_dd = dd.io.from_dask_array(dataArray_da,
                                             columns=['c_bar',
                                                      'Xi_iso_0.5',
                                                      'omega_model',
                                                      'omega_DNS_filtered',
                                                      'grad_DNS_c_05',
                                                      'dirac_05'])

        # filter the data set and remove unecessary entries
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] > 0.01]
        self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] < 0.99]


        # remove all Xi_iso < 1e-2 from the stored data set -> less memory
        self.dataArray_dd = self.dataArray_dd[(self.dataArray_dd['Xi_iso_0.5'] > 1e-2) ]

        print('Xi_iso < 1 are included!')

        print('Computing data array ...')
        self.dataArray_df = self.dataArray_dd.sample(frac=0.2).compute()

        print('Writing output to csv ...')
        self.dataArray_df.to_csv(filename,index=False)
        print('Data has been written.\n\n')

    def plot_histograms(self, c_tilde, this_rho_c_reshape, this_rho_reshape, this_RR_reshape_DNS):
        return NotImplementedError

    def plot_histograms_intervals(self,c_tilde,this_rho_c_reshape,this_rho_reshape,c_bar,this_RR_reshape_DNS,wrinkling=1):
        return NotImplementedError




###################################################################
# NEW CLASS WITH PFITZNERS WRINKLING FACTOR
# THEREs a difference between xi and Xi!!! see the paper...

class dns_analysis_dirac_xi(dns_analysis_dirac):
    '''
    computes the numerical delta function in xi space instead of c-space
    --> more accurate!
    '''

    def __init__(self, case, eps_factor=100, c_iso_values=[0.85]):
        # extend super class with additional input parameters
        super(dns_analysis_dirac_xi, self).__init__(case)
        self.c_iso_values = c_iso_values
        self.eps_factor = eps_factor
        # the convolution of dirac x grad_c at all relevant iso values and LES filtered will be stored here
        self.Xi_iso_filtered = np.zeros((self.Nx,self.Ny,self.Nz,len(self.c_iso_values)))

        self.xi_np = None        # to be filled
        self.xi_iso_values = None   # converted c_iso_values into xi-space
        self.grad_xi_DNS = None     # xi-Field gradients on DNS mesh
        self.dirac_times_grad_xi = None

        #print('You are using the Dirac version...')

    def convert_c_field_to_xi(self):
        '''
        converts the c-field to the xi field (Eq. 13, Pfitzner, FTC, 2019)
        :return:
        '''

        c_clipped = self.c_data_np*0.9999 + 1e-5

        self.xi_np = 1/self.m * np.log(c_clipped**self.m/ (1 - c_clipped**self.m) )
        self.xi_iso_values = [1/self.m * np.log(c**self.m/ (1 - c**self.m) ) for c in self.c_iso_values] #is a list


    def convert_c_to_xi(self,c):
        '''
        converts the c value to the xi value (Eq. 13, Pfitzner, FTC, 2019)
        works for arrays of any size
        :return:
        '''

        c_clipped = c*0.9999 + 1e-5

        xi = 1/self.m * np.log(c_clipped**self.m/ (1 - c_clipped**self.m) )
        return xi


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
        X_vec = X.reshape(self.Nx*self.Ny*self.Nz)

        dirac_vec = np.zeros(len(X_vec))

        # Fallunterscheidung für X < 0
        for id, x in enumerate(X_vec):
            if x < 1:
                dirac_vec[id] =1/(2*eps) * (1 + np.cos(np.pi * x)) * self.delta_x
                #print('dirac_vec: ',dirac_vec[id])

        dirac_array = dirac_vec.reshape(self.Nx,self.Ny,self.Nz)

        return dirac_array


    def compute_Xi_iso_dirac_xi_old(self):

        # check if self.grad_c_DNS was computed, if not -> compute it
        if self.grad_xi_DNS is None:
            self.compute_DNS_grad_xi_4thO()

        # loop over the different c_iso values
        for id, xi_iso in enumerate(self.xi_iso_values):

            xi_phi = self.compute_phi_xi(xi_iso)
            dirac_xi = self.compute_dirac_cos(xi_phi)
            self.dirac_times_grad_xi = (dirac_xi * self.grad_xi_DNS).reshape(self.Nx,self.Ny,self.Nz)
            #print('dirac_times_grad_xi: ', self.dirac_times_grad_xi)

            # check if integral is 1 for arbitrary
            # Line integral
            print('line integrals over dirac_times_grad(c):')
            # print(np.trapz(dirac_times_grad_c[:, 250, 250]))
            # print(np.trapz(dirac_times_grad_c[250, :, 250]))
            print(np.trapz(self.dirac_times_grad_xi[250, 250, :]))

            # TODO: check if that is correct!
            dirac_LES_sums = self.compute_LES_cell_sum(self.dirac_times_grad_xi)
            self.Xi_iso_filtered[:, :, :, id] = dirac_LES_sums / self.filter_width**2

    # overrides main method
    def run_analysis_dirac(self,filter_width ,filter_type, c_analytical=False):
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
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

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

        # compute Xi iso surface area for all c_iso values
        self.compute_Xi_iso_dirac_xi_old()

        # creat dask array and reshape all data
        # a bit nasty for list in list as of variable c_iso values
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.Xi_iso_filtered[:,:,:,0].reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.omega_model_cbar.reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.omega_DNS_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                    #self.grad_c_05.reshape(self.Nx*self.Ny*self.Nz,1),
                                    #self.dirac_05.reshape(self.Nx*self.Ny*self.Nz, 1)
                                  ])

        filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_dirac_xi.csv')

        self.dataArray_dd = dd.io.from_dask_array(dataArray_da,
                                             columns=['c_bar',
                                                      'Xi_iso_0.5',
                                                      'omega_model',
                                                      'omega_DNS_filtered',
                                                      #'grad_DNS_c_05',
                                                      #'dirac_05'
                                                        ])

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


    # def compute_LES_cell_sum(self,input_array):
    #     # get the sum of values inside an LES cell
    #     print('computing cell sum...')
    #
    #     try:
    #         assert len(input_array.shape) == 3
    #     except AssertionError:
    #         print('input array must be 3D!')
    #
    #     output_array = copy.copy(input_array)
    #
    #     output_array *= 0.0                 # set output array to zero
    #
    #     half_filter = int(self.filter_width / 2)
    #
    #     for l in range(half_filter, self.Nx - half_filter, 1):
    #         for m in range(half_filter, self.Nx - half_filter, 1):
    #             for n in range(half_filter, self.Nx - half_filter, 1):
    #
    #                 this_LES_box = (input_array[l - half_filter: l + half_filter,
    #                                 m - half_filter: m + half_filter,
    #                                 n - half_filter: n + half_filter])
    #
    #                 # compute c_bar of current LES box
    #                 output_array[l,m,n] = this_LES_box.sum()
    #
    #     return output_array

    def compute_DNS_grad_xi(self):
        '''
        computes the flame surface area in the DNS based on gradients of xi of neighbour cells
        magnitude of the gradient:
        :return: abs(grad(Xi))
        '''

        print('Computing DNS gradients in xi space 2nd Order ...')

        # check if self.xi_np is filled
        if self.xi_np is None:
            self.convert_c_field_to_xi()

        # create empty array
        grad_xi_DNS = np.zeros([self.Nx,self.Ny,self.Nz])

        # compute gradients from the boundaries away ...
        for l in range(1,self.Nx-1):
            for m in range(1,self.Ny-1):
                for n in range(1,self.Nz-1):
                    this_DNS_gradX = (self.xi_np[l+1, m, n] - self.xi_np[l-1,m, n])/(2 * self.delta_x)
                    this_DNS_gradY = (self.xi_np[l, m+1, n] - self.xi_np[l, m-1, n]) / (2 * self.delta_x)
                    this_DNS_gradZ = (self.xi_np[l, m, n+1] - self.xi_np[l, m, n-1]) / (2 * self.delta_x)
                    # compute the magnitude of the gradient
                    this_DNS_magGrad_c = np.sqrt(this_DNS_gradX**2 + this_DNS_gradY**2 + this_DNS_gradZ**2)

                    grad_xi_DNS[l,m,n] = this_DNS_magGrad_c

        return grad_xi_DNS

    def compute_DNS_grad_xi_4thO(self):
        '''
        computes the flame surface area in the DNS based on gradients of xi of neighbour cells
        magnitude of the gradient:
        :return: abs(grad(Xi))
        '''

        print('Computing DNS gradients in xi space 4th Order ...')

        # check if self.xi_np is filled
        if self.xi_np is None:
            self.convert_c_field_to_xi()

        # create empty array
        grad_xi_DNS = np.zeros([self.Nx,self.Ny,self.Nz])

        # compute gradients from the boundaries away ...
        for l in range(1,self.Nx-2):
            for m in range(1,self.Ny-2):
                for n in range(1,self.Nz-2):
                    this_DNS_gradX = (-self.xi_np[l + 2, m, n] + 8 * self.xi_np[l + 1, m, n] - 8 *
                                      self.xi_np[l - 1, m, n] + self.xi_np[l - 2, m, n]) / (12 * self.delta_x)

                    this_DNS_gradY = (-self.xi_np[l, m + 2, n] + 8 * self.xi_np[l, m + 1, n] - 8 *
                                      self.xi_np[l, m - 1, n] + self.xi_np[l, m - 2, n]) / (12 * self.delta_x)

                    this_DNS_gradZ = (-self.xi_np[l, m, n + 2] + 8 * self.xi_np[l, m, n + 1] - 8 *
                                      self.xi_np[l, m, n - 1] + self.xi_np[l, m, n - 2]) / (12 * self.delta_x)

                    # compute the magnitude of the gradient
                    this_DNS_magGrad_c = np.sqrt(this_DNS_gradX**2 + this_DNS_gradY**2 + this_DNS_gradZ**2)

                    grad_xi_DNS[l,m,n] = this_DNS_magGrad_c

        return grad_xi_DNS


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

class dns_analysis_dirac_FSD(dns_analysis_dirac_xi):
    # new implementation with numerical delta dirac function

    def __init__(self,case, eps_factor=100,c_iso_values=[0.01,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.99]):
        # extend super class with additional input parameters
        super(dns_analysis_dirac_FSD, self).__init__(case, eps_factor,c_iso_values)

        # number of c_iso slices
        self.N_c_iso = len(self.c_iso_values)

        # this is a 4D array to store all dirac values for the different c_iso values
        self.dirac_xi_fields = np.zeros([self.N_c_iso,self.Nx,self.Ny,self.Nz])

        # set up a new omega_bar field to store the exact solution from the pdf...no real name yet
        self.omega_model_exact = np.zeros([self.Nx,self.Ny,self.Nz])

        #print('You are using the Dirac version...')
        print('You are using the new FSD routine...')

    def compute_dirac_xi_iso_fields(self):

        # compute the xi field
        # self.xi_np and self.xi_iso_values (from self.c_iso_values)
        self.convert_c_field_to_xi()

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
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

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
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.omega_model_exact.reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.omega_DNS_filtered.reshape(self.Nx*self.Ny*self.Nz,1)
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

class dns_analysis_dirac_FSD_alt(dns_analysis_dirac_xi):
    # new implementation with numerical delta dirac function

    def __init__(self,case, eps_factor=100,
                 c_iso_values=[0.001, 0.3,0.5,0.55,0.6,0.65,0.7,0.725,0.75,0.775,0.8,0.82,0.84,0.85,0.86,0.88,0.9,0.92,0.94,0.98 ,0.999]):
        # extend super class with additional input parameters
        super(dns_analysis_dirac_FSD_alt, self).__init__(case, eps_factor,c_iso_values)

        # number of c_iso slices
        self.N_c_iso = len(self.c_iso_values)

        # this is a 4D array to store all dirac values for the different c_iso values
        self.dirac_xi_fields = np.zeros([self.N_c_iso,self.Nx,self.Ny,self.Nz])

        # set up a new omega_bar field to store the exact solution from the pdf...no real name yet
        self.omega_model_exact = np.zeros([self.Nx,self.Ny,self.Nz])

        print('You are using the new alternative FSD routine...')


    def compute_dirac_xi_iso_fields(self):
        '''
        compute the xi field
        self.xi_np and self.xi_iso_values (from self.c_iso_values)
        :return:
        '''

        self.convert_c_field_to_xi()

        # loop over the different c_iso values
        for id, xi_iso in enumerate(self.xi_iso_values):

            print('Computing delta_dirac in xi-space for c_iso=%f'%self.c_iso_values[id])
            xi_phi = self.compute_phi_xi(xi_iso)
            dirac_xi = self.compute_dirac_cos(xi_phi)

            #write each individual dirac_xi field into the storage array
            self.dirac_xi_fields[id,:,:,:] = dirac_xi


    def compute_omega_FSD(self):
        # '''
        # omega computation with FSD method according to Pfitzner
        # omega_bar = \int_0^1 (m+1)*c_iso**m 1/Delta_LES**3 \int_cell \delta(\xi(x) - \xi_iso) dx dc_iso
        # :return: NULL
        # :param self.omega_model_exact is the FSD omgea
        # '''

        print('Computing exact omega bar ...')

        self.compute_dirac_xi_iso_fields()

        # convolute the iso fields wiht omega/dc/dxi
        for id, this_c_iso in enumerate(self.c_iso_values):

            omega_over_grad_c = self.compute_omega_over_grad_c(this_c_iso)
            self.dirac_xi_fields[id,:,:,:] = self.dirac_xi_fields[id,:,:,:] * omega_over_grad_c #ACHTUNG: überschreibt self.dirac_xi_fields!

        # do simpson integration over the whole range of c_iso values
        omega_integrated = simps(self.dirac_xi_fields,self.c_iso_values,axis=0)

        try:
            assert omega_integrated == (self.Nx,self.Ny,self.Nz)
        except AssertionError:
            print('omega_integrant shape', omega_integrated.shape)

        # Works fast and is correct!
        print('Top Hat filtering to get omega_model_exact')
        self.omega_model_exact  = self.apply_filter(omega_integrated)/(self.Delta_LES/self.filter_width)**3

        # free some memory, array is not needed anymore case
        del self.dirac_xi_fields


    def compute_omega_over_grad_c(self,c_iso):
        '''
        Computes the analytical expression for \omega(c*)/|dc*/dx|
        has to be computed for each c_iso value indivdually
        :return:
        '''

        return (self.m + 1)* c_iso** self.m


    def compute_phi_c(self,c_iso):
        '''
        computes the difference between abs(c(x)-c_iso)
        see Pfitzner notes Eq. 3
        :param c_iso: np vector (1D np.array)
        :return: 1D np.array
        '''

        c_data_vec = self.c_data_np.reshape(self.Nx*self.Ny*self.Nz)

        return abs(c_data_vec - c_iso)


    def compute_Xi_iso_dirac_c(self,c_iso):
        '''

        :param c_iso: 3D or 1D np.array
        :return: 3D np.array of the Xi field
        '''

        # make sure c_iso is a vector!
        if np.ndim(c_iso) > 1:
            c_iso = c_iso.reshape(self.Nx*self.Ny*self.Nz)

        # check if self.grad_c_DNS was computed, if not -> compute it
        if self.grad_c_DNS is None:
            self.grad_c_DNS = self.compute_DNS_grad_4thO()

        grad_c_DNS_vec = self.grad_c_DNS.reshape(self.Nx*self.Ny*self.Nz)

        c_phi = self.compute_phi_c(c_iso)
        dirac = self.compute_dirac_cos(c_phi)
        dirac_vec = dirac.reshape(self.Nx*self.Ny*self.Nz)

        dirac_times_grad_c_vec = (dirac_vec * grad_c_DNS_vec)

        #################################
        # print('dirac_imes_grad_c: ', dirac_times_grad_c)

        # convert from vector back to 3D array
        dirac_times_grad_c_arr = dirac_times_grad_c_vec.reshape(self.Nx,self.Ny,self.Nz)

        dirac_LES_filter = self.apply_filter(dirac_times_grad_c_arr) #self.compute_LES_cell_sum(dirac_times_grad_c_arr)

        Xi_iso_filtered = dirac_LES_filter * (self.filter_width**3 / self.filter_width**2)  # Conversion to Xi-space!

        # dirac_LES_sums = self.compute_LES_cell_sum(self.dirac_times_grad_xi)
        # self.Xi_iso_filtered[:, :, :, id] = dirac_LES_sums / self.filter_width ** 2

        return Xi_iso_filtered


    def compute_Xi_iso_dirac_xi(self,c_iso):
        '''
        use the filter function to get the cell integral of dirac values \int_\Omeaga
        :param c_iso: 3D or 1D np.array
        :return: 3D np.array of the Xi field and dirac_times_grad_xi_arr
        '''

        # make sure c_iso is a vector!
        if np.ndim(c_iso) > 1:
            c_iso = c_iso.reshape(self.Nx*self.Ny*self.Nz)

        # converts the c field to xi field: self.xi_np
        self.convert_c_field_to_xi()

        # compute xi_iso
        xi_iso = self.convert_c_to_xi(c=c_iso)

        xi_phi = self.compute_phi_xi(xi_iso)
        dirac_xi = self.compute_dirac_cos(xi_phi)

        # check if self.grad_c_DNS was computed, if not -> compute it
        if self.grad_xi_DNS is None:
            self.grad_xi_DNS = self.compute_DNS_grad_xi_4thO()

        grad_xi_DNS_vec = self.grad_xi_DNS.reshape(self.Nx*self.Ny*self.Nz)

        dirac_xi_vec = dirac_xi.reshape(self.Nx*self.Ny*self.Nz)

        dirac_times_grad_xi_vec = (dirac_xi_vec * grad_xi_DNS_vec)

        # convert from vector back to 3D array
        dirac_times_grad_xi_arr = dirac_times_grad_xi_vec.reshape(self.Nx,self.Ny,self.Nz)

        dirac_LES_filter = self.apply_filter(dirac_times_grad_xi_arr) #self.compute_LES_cell_sum(dirac_times_grad_c_arr)

        Xi_iso_filtered = dirac_LES_filter * (self.filter_width)  # Conversion to Xi-space!

        # dirac_LES_sums = self.compute_LES_cell_sum(self.dirac_times_grad_xi)
        # self.Xi_iso_filtered[:, :, :, id] = dirac_LES_sums / self.filter_width ** 2

        return Xi_iso_filtered, dirac_times_grad_xi_arr


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
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

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
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.omega_model_exact.reshape(self.Nx*self.Ny*self.Nz,1),
                                    self.omega_DNS_filtered.reshape(self.Nx*self.Ny*self.Nz,1)
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

class dns_analysis_dirac_compare(dns_analysis_dirac_FSD_alt):
    ''' compares the different Sigma computations
        1. based on slices and convolution with corresponding \omega (FSD assumption)
        2. based on c_iso = 0.85
        3. based on c_iso = c_bar
    '''

    def __init__(self,case, eps_factor,c_iso_values):
        # extend super class with additional input parameters
        super(dns_analysis_dirac_compare, self).__init__(case, eps_factor,c_iso_values)

        print('This is the version to compare the different Sigmas ...')


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
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
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

        # 4th method: computes wrinkling factor
        self.get_wrinkling(order='4th')

        # prepare data for output to csv file
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                  self.omega_model_exact.reshape(self.Nx*self.Ny*self.Nz,1),
                                  self.omega_DNS_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                  self.omega_model_cbar.reshape(self.Nx*self.Ny*self.Nz,1),
                                  Xi_iso_085.reshape(self.Nx*self.Ny*self.Nz,1),
                                  Xi_iso_cbar.reshape(self.Nx*self.Ny*self.Nz,1),
                                  self.wrinkling_factor.reshape(self.Nx*self.Ny*self.Nz,1)
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


    def run_analysis_dirac_FSD(self,filter_width ,filter_type, c_analytical=False):
        return NotImplementedError


################################################
# Analyse the 40 slices from UPRIME15
# 27.3.2020

class dns_analysis_compare_40slices(dns_analysis_dirac_FSD_alt):
    '''
        Analyse the reduced data set for comparison with Pfitzners implementation
    '''

    def __init__(self,case, eps_factor,c_iso_values):
        # extend super class with additional input parameters
        super(dns_analysis_compare_40slices, self).__init__(case, eps_factor,c_iso_values)

        print('This is the version to analyse the 40 slices ...')

    def read_in_c_40slices(self):
        '''
        read in the 40 slices data which is stored as .npy
        :return:
        '''

        self.c_data_np = np.load(self.c_path+'.npy')

        self.Nx = self.c_data_np.shape[0]
        self.Ny = self.c_data_np.shape[1]
        self.Nz = self.c_data_np.shape[2]

        print('Shape of c_data_np: ',self.c_data_np.shape)

        # setup new, as Nx is not same as Nz!
        self.Xi_iso_filtered = np.zeros((self.Nx, self.Ny, self.Nz, len(self.c_iso_values)))

    # def run_analysis_dirac_40slices(self,filter_width ,filter_type, c_analytical=False):
    #     '''
    #     :param filter_width: DNS points to filter
    #     :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
    #     :param c_analytical: compute c minus analytically
    #     :return:
    #     '''
    #     # run the analysis and compute the wrinkling factor -> real 3D cases
    #     # interval is like nth point, skips some nodes
    #     self.filter_type = filter_type
    #
    #     print('You are using %s filter!' % self.filter_type)
    #
    #     self.filter_width = int(filter_width)
    #
    #     self.c_analytical = c_analytical
    #     if self.c_analytical is True:
    #         print('You are using Hypergeometric function for c_minus (Eq.35)!')
    #
    #     # filter the c and rho field
    #     print('Filtering c field ...')
    #     #self.rho_filtered = self.apply_filter(self.rho_data_np)
    #     self.c_filtered = self.apply_filter(self.c_data_np)
    #
    #     # Compute the scaled Delta (Pfitzner PDF)
    #     self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
    #     print('Delta_LES is: %.3f' % self.Delta_LES)
    #     flame_thickness = self.compute_flamethickness()
    #     print('Flame thickness: ',flame_thickness)
    #
    #     #maximum possible wrinkling factor
    #     print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)
    #
    #     # compute self.omega_model_cbar: the Pfitzner model for the planar flame
    #     self.compute_Pfitzner_model()
    #
    #     # # 1st method
    #     # # compute omega_model_exact with FSD from Pfitzner
    #     # self.compute_omega_FSD()
    #
    #     # 2nd method
    #     Xi_iso_085 = self.compute_Xi_iso_dirac_c(c_iso=0.85)
    #
    #     # # 3rd method
    #     # Xi_iso_cbar = self.compute_Xi_iso_dirac_c(c_iso=self.c_filtered)
    #     #
    #     # # 4th method: computes wrinkling factor
    #     # self.get_wrinkling(order='4th')
    #
    #     # prepare data for output to csv file
    #     dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
    #                               #self.omega_model_exact.reshape(self.Nx*self.Ny*self.Nz,1),
    #                               self.omega_DNS_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
    #                               self.omega_model_cbar.reshape(self.Nx*self.Ny*self.Nz,1),
    #                               Xi_iso_085.reshape(self.Nx*self.Ny*self.Nz,1),
    #                               #Xi_iso_cbar.reshape(self.Nx*self.Ny*self.Nz,1),
    #                               #self.wrinkling_factor.reshape(self.Nx*self.Ny*self.Nz,1)
    #                               ])
    #
    #     filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_compare.csv')
    #
    #     self.dataArray_dd = dd.io.from_dask_array(dataArray_da,columns=
    #                                               ['c_bar',
    #                                                'omega_DNS_filtered',
    #                                                'omega_model_planar',
    #                                                'Xi_iso_085'
    #                                                ])
    #
    #     # # filter the data set and remove unecessary entries
    #     # self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] > 0.01]
    #     # self.dataArray_dd = self.dataArray_dd[self.dataArray_dd['c_bar'] < 0.99]
    #
    #     print('Computing data array ...')
    #     self.dataArray_df = self.dataArray_dd.compute()
    #
    #     print('Writing output to csv ...')
    #     self.dataArray_df.to_csv(filename,index=False)
    #     print('Data has been written.\n\n')


    def run_analysis_compare_Xi(self,filter_width ,filter_type, c_analytical=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        self.every_nth = 1

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        # filter the c and rho field
        print('Filtering c field ...')
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

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

        # # compute Xi iso surface area for all c_iso values
        # self.compute_Xi_iso_dirac_xi_old()

        # 2nd method to compute Xi
        Xi_iso_085, dirac_grad_xi_arr = self.compute_Xi_iso_dirac_xi(c_iso=0.85)

        # marching cubes to compute Xi
        isoArea_coefficient = self.compute_isoArea(c_iso=0.85)

        # creat dask array and reshape all data
        # a bit nasty for list in list as of variable c_iso values
        dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx*self.Ny*self.Nz,1),
                                  self.omega_DNS_filtered.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  self.omega_model_cbar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  #self.Xi_iso_filtered[:,:,:,0].reshape(self.Nx*self.Ny*self.Nz,1),
                                  Xi_iso_085.reshape(self.Nx*self.Ny*self.Nz,1),
                                  isoArea_coefficient.reshape(self.Nx*self.Ny*self.Nz,1),
                                  dirac_grad_xi_arr.reshape(self.Nx*self.Ny*self.Nz,1),
                                  ])

        filename = join(self.case, 'filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_compare_xi.csv')

        self.dataArray_dd = dd.io.from_dask_array(dataArray_da,columns=
                                                  ['c_bar',
                                                   'omega_DNS_filtered',
                                                   'omega_model_planar',
                                                   'Xi_iso_085',
                                                   'Xi_march_cube_085',
                                                   'dirac_grad_xi'
                                                   ])

        print('Computing data array ...')
        self.dataArray_df = self.dataArray_dd.compute()

        print('Writing output to csv ...')
        self.dataArray_df.to_csv(filename,index=False)
        print('Data has been written.\n\n')



################################################
# INCLUDE NOW THE VELOCITY DATA UWV.dat to compute sl
# 20.4.2020

class dns_analysis_UVW(dns_analysis_dirac_FSD_alt):
    '''
        Analysis includes now the velocity data
    '''

    def __init__(self,case, eps_factor,c_iso_values):
        # extend super class with additional input parameters
        super(dns_analysis_UVW, self).__init__(case, eps_factor,c_iso_values)

        print('This is the version that includes velocity components ...')

        self.U = None
        self.V = None
        self.W = None

        self.U_bar = None
        self.V_bar = None
        self.W_bar = None

        self.U_prime = None
        self.V_prime = None
        self.W_prime = None

        # create empty array
        self.grad_U_bar = np.zeros([self.Nx, self.Ny, self.Nz])
        self.grad_V_bar = np.zeros([self.Nx, self.Ny, self.Nz])
        self.grad_W_bar = np.zeros([self.Nx, self.Ny, self.Nz])

    # constructor end


    def read_UVW(self):
        '''
        reads in the velocity fields and stores it as np.array. 3D!!
        :return: nothing
        '''

        print('Reading the velocity data ...')

        U_np = pd.read_csv(join(self.case, 'U.dat'), names=['U']).values.astype(dtype=np.float32)
        V_np = pd.read_csv(join(self.case, 'V.dat'), names=['U']).values.astype(dtype=np.float32)
        W_np = pd.read_csv(join(self.case, 'W.dat'), names=['U']).values.astype(dtype=np.float32)

        self.U = U_np.reshape(self.Nx,self.Ny,self.Nz)
        self.V = V_np.reshape(self.Nx,self.Ny,self.Nz)
        self.W = W_np.reshape(self.Nx,self.Ny,self.Nz)

        # delete to save memory
        del U_np, V_np, W_np

    #@jit(nopython=True, parallel=True)
    def compute_U_prime(self):
        '''
        # compute the SGS velocity fluctuations
        # u_prime is the STD of the U-DNS components within a LES cell
        :return: nothing
        '''
        print('Computing U prime')
        self.U_prime = scipy.ndimage.generic_filter(self.U, np.var, mode='wrap',
                                                    size=(self.filter_width, self.filter_width,self.filter_width))
        print('Computing V prime')
        self.V_prime = scipy.ndimage.generic_filter(self.V, np.var, mode='wrap',
                                                    size=(self.filter_width, self.filter_width, self.filter_width))
        print('Computing W prime')
        self.W_prime = scipy.ndimage.generic_filter(self.W, np.var, mode='wrap',
                                                    size=(self.filter_width, self.filter_width, self.filter_width))

    def compute_U_prime_alternative(self,U,V,W,Nx,Ny,Nz, filter_width):

        # translate that to CYTHON

        # print('\noutput_array.shape: ',output_array.shape)

        output_U = np.zeros((Nx, Ny, Nz))                 # set output array to zero
        output_V = np.zeros((Nx, Ny, Nz))
        output_W = np.zeros((Nx, Ny, Nz))

        half_filter = int(filter_width / 2)

        les_filter = filter_width*filter_width*filter_width

        for l in range(half_filter, Nx - half_filter, 1):
            for m in range(half_filter, Ny - half_filter, 1):
                for n in range(half_filter, Nz - half_filter, 1):

                    this_LES_box_U = (U[l - half_filter: l + half_filter,
                                    m - half_filter: m + half_filter,
                                    n - half_filter: n + half_filter])

                    this_LES_box_V = (V[l - half_filter: l + half_filter,
                                    m - half_filter: m + half_filter,
                                    n - half_filter: n + half_filter])

                    this_LES_box_W = (W[l - half_filter: l + half_filter,
                                    m - half_filter: m + half_filter,
                                    n - half_filter: n + half_filter])

                    # compute the means of each cell
                    mean_U = 0.0
                    mean_V = 0.0
                    mean_W = 0.0
                    this_U_prime = 0.0
                    this_V_prime = 0.0
                    this_W_prime = 0.0
                    for i in range(0,filter_width):
                        for j in range(0, filter_width):
                            for k in range(0, filter_width):
                                mean_U = mean_U + this_LES_box_U[i, j, k]
                                mean_V = mean_V + this_LES_box_V[i, j, k]
                                mean_W = mean_W + this_LES_box_W[i, j, k]

                    mean_U = mean_U / les_filter
                    mean_V = mean_V / les_filter
                    mean_W = mean_W / les_filter
                    # compute the variance of each cell

                    for i in range(0,filter_width):
                        for j in range(0, filter_width):
                            for k in range(0, filter_width):
                                this_U_prime = this_U_prime+ (this_LES_box_U[i,j,k] - mean_U) *\
                                               (this_LES_box_U[i,j,k] - mean_U)
                                this_V_prime = this_V_prime + (this_LES_box_V[i, j, k] - mean_V) * \
                                               (this_LES_box_V[i, j, k] - mean_V)
                                this_W_prime = this_W_prime + (this_LES_box_W[i, j, k] - mean_W) * \
                                               (this_LES_box_W[i, j, k] - mean_W)

                    # compute c_bar of current LES box
                    output_U[l,m,n] = this_U_prime
                    output_V[l, m, n] = this_V_prime
                    output_W[l, m, n] = this_W_prime

        return output_U, output_W, output_W

    # def compute_Ka_sgs(self):
    #     '''
    #     # compute the SGS Ka number
    #     :return: nothing
    #     '''
    #
    #     if self.case.startswith('NX512'):
    #         self.d_th_DNS = 0.051
    #     elif self.case.startswith('NX768'):
    #         self.d_th_DNS = 0.042
    #     else:
    #         print('I dont know which d_th_DNS!')
    #         sys.exit()
    #
    #     self.s_L =
    #     print('computing Ka_sgs with hard coded s_L=%f and d_th=%f' % (s_L,))

    #@jit(nopython=True, parallel=True)
    def compute_gradU_LES_4thO(self):
        # '''
        # Compute the magnitude of the gradient of the DNS c-field, based on neighbour cells
        # 4th Order central differencing
        # :return: nothing
        # '''

        # TODO: externalize? -> DONE!

        print('Computing gradients of U_bar on DNS mesh 4th Order...')

        # compute gradients from the boundaries away ...
        for l in range(2,self.Nx-2):
            for m in range(2,self.Ny-2):
                for n in range(2,self.Nz-2):
                    this_U_gradX = (-self.U_bar[l+2, m, n] + 8*self.U_bar[l+1,m, n] - 8*self.U_bar[l-1,m, n] + self.U_bar[l-2, m, n])/(12 * self.delta_x)
                    this_U_gradY = (-self.U_bar[l, m+2, n] + 8*self.U_bar[l,m+1, n] - 8*self.U_bar[l,m-1, n] + self.U_bar[l, m-2, n])/(12 * self.delta_x)
                    this_U_gradZ = (-self.U_bar[l, m, n+2] + 8*self.U_bar[l,m, n+1] - 8*self.U_bar[l,m, n-1] + self.U_bar[l, m, n+2])/(12 * self.delta_x)

                    this_V_gradX = (-self.V_bar[l+2, m, n] + 8*self.V_bar[l+1,m, n] - 8*self.V_bar[l-1,m, n] + self.V_bar[l-2, m, n])/(12 * self.delta_x)
                    this_V_gradY = (-self.V_bar[l, m+2, n] + 8*self.V_bar[l,m+1, n] - 8*self.V_bar[l,m-1, n] + self.V_bar[l, m-2, n])/(12 * self.delta_x)
                    this_V_gradZ = (-self.V_bar[l, m, n+2] + 8*self.V_bar[l,m, n+1] - 8*self.V_bar[l,m, n-1] + self.V_bar[l, m, n+2])/(12 * self.delta_x)

                    this_W_gradX = (-self.W_bar[l+2, m, n] + 8*self.W_bar[l+1,m, n] - 8*self.W_bar[l-1,m, n] + self.W_bar[l-2, m, n])/(12 * self.delta_x)
                    this_W_gradY = (-self.W_bar[l, m+2, n] + 8*self.W_bar[l,m+1, n] - 8*self.W_bar[l,m-1, n] + self.W_bar[l, m-2, n])/(12 * self.delta_x)
                    this_W_gradZ = (-self.W_bar[l, m, n+2] + 8*self.W_bar[l,m, n+1] - 8*self.W_bar[l,m, n-1] + self.W_bar[l, m, n+2])/(12 * self.delta_x)

                    # compute the magnitude of the gradient
                    this_magGrad_U = np.sqrt(this_U_gradX ** 2 + this_U_gradY ** 2 + this_U_gradZ ** 2)
                    this_magGrad_V = np.sqrt(this_V_gradX ** 2 + this_V_gradY ** 2 + this_V_gradZ ** 2)
                    this_magGrad_W = np.sqrt(this_W_gradX ** 2 + this_W_gradY ** 2 + this_W_gradZ ** 2)

                    self.grad_U_bar[l, m, n] = this_magGrad_U
                    self.grad_V_bar[l, m, n] = this_magGrad_V
                    self.grad_W_bar[l, m, n] = this_magGrad_W


    def run_analysis_UVW(self,filter_width ,filter_type, c_analytical=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        ##Todo: muss noch überarbeitet werden!

        self.every_nth = 1

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        #('Reading in U fieds ...')
        self.read_UVW()

        # filter the c and rho field
        print('Filtering c field ')
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)

        #Todo: In paralllel? How?
        print('Filtering U components')
        self.U_bar = self.apply_filter(self.U)
        self.V_bar = self.apply_filter(self.V)
        self.W_bar = self.apply_filter(self.W)

        ###########################
        print('Computing U primes')
        # imported cython function: dtype needs to be float32!
        U_prime, V_prime, W_prime = compute_U_prime_cython(np.ascontiguousarray(self.U, dtype=np.float32),
                                                            np.ascontiguousarray(self.V, dtype=np.float32),
                                                            np.ascontiguousarray(self.W, dtype=np.float32),
                                                            self.Nx, self.Ny, self.Nz, self.filter_width)

        #  CONVERT BACK TO NUMPY ARRAY FROM CYTHON
        self.U_prime = np.asarray(U_prime)
        self.V_prime = np.asarray(V_prime)
        self.W_prime = np.asarray(W_prime)



        # free memory and delete U fields
        del self.U, self.W, self.V, U_prime, V_prime, W_prime

        ###########################
        # compute the magnitude of gradients of the velocity field
        #self.compute_gradU_LES_4thO()
        # use imported cython function: dtype needs to be float32!
        grad_U_bar, grad_V_bar, grad_W_bar = compute_gradU_LES_4thO_cython(
                                                                    np.ascontiguousarray(self.U_bar, dtype=np.float32),
                                                                    np.ascontiguousarray(self.V_bar, dtype=np.float32),
                                                                    np.ascontiguousarray(self.W_bar, dtype=np.float32),
                                                                    self.Nx, self.Ny, self.Nz, np.float32(self.delta_x))

        #  CONVERT BACK TO NUMPY ARRAY FROM CYTHON
        self.grad_U_bar = np.asarray(grad_U_bar)
        self.grad_V_bar = np.asarray(grad_V_bar)
        self.grad_W_bar = np.asarray(grad_W_bar)

        del grad_U_bar, grad_V_bar, grad_W_bar  #free memory

        ###########################
        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)


        ###########################
        # compute abs(grad(xi)) on the whole DNS domain
        self.convert_c_field_to_xi()
        # self.grad_xi_DNS = self.compute_DNS_grad_xi_4thO()
        # CYTHON
        print('Computing gradients of xi on DNS mesh 4th Order with Cython')
        grad_xi_DNS = compute_DNS_grad_4thO_cython(np.ascontiguousarray(self.xi_np, dtype=np.float32),
                                                      self.Nx, self.Ny, self.Nz, np.float32(self.delta_x))

        self.grad_xi_DNS = np.asarray(grad_xi_DNS)

        del grad_xi_DNS     # free memory


        ###########################
        # compute abs(grad(xi)) on the whole DNS domain
        # CYTHON
        print('Computing gradients of c_bar on DNS mesh 4th Order with Cython')
        grad_c_bar_DNS = compute_DNS_grad_4thO_cython(np.ascontiguousarray(self.c_filtered, dtype=np.float32),
                                                      self.Nx, self.Ny, self.Nz, np.float32(self.delta_x))

        grad_c_bar_DNS = np.asarray(grad_c_bar_DNS)

        #del grad_xi_DNS     # free memory

        ###########################
        # compute omega based on pfitzner
        self.compute_Pfitzner_model()

        # 2nd method to compute Xi
        Xi_iso_085, dirac_grad_xi_arr = self.compute_Xi_iso_dirac_xi(c_iso=0.85)

        # #creat dask array and reshape all data

        self.dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx * self.Ny * self.Nz,1),
                                  grad_c_bar_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  self.omega_DNS_filtered.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  self.omega_model_cbar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  Xi_iso_085.reshape(self.Nx * self.Ny * self.Nz,1),
                                  self.U_bar.reshape(self.Nx * self.Ny * self.Nz,1),
                                  self.V_bar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  self.W_bar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  self.U_prime.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  self.V_prime.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  self.W_prime.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  self.grad_U_bar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  self.grad_V_bar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                  self.grad_W_bar.reshape(self.Nx * self.Ny * self.Nz, 1)
                                  ])


        # self.dataArray_da = da.hstack([self.c_filtered[200:300,200:300,200:300].reshape(100**3,1),
        #                           grad_c_bar_DNS[200:300,200:300,200:300].reshape(100**3, 1),
        #                           self.omega_DNS_filtered[200:300,200:300,200:300].reshape(100**3, 1),
        #                           self.omega_model_cbar[200:300,200:300,200:300].reshape(100**3, 1),
        #                           Xi_iso_085[200:300,200:300,200:300].reshape(100**3,1),
        #                           self.U_bar[200:300,200:300,200:300].reshape(100**3,1),
        #                           self.V_bar[200:300,200:300,200:300].reshape(100**3, 1),
        #                           self.W_bar[200:300,200:300,200:300].reshape(100**3, 1),
        #                           self.U_prime[200:300,200:300,200:300].reshape(100**3, 1),
        #                           self.V_prime[200:300,200:300,200:300].reshape(100**3, 1),
        #                           self.W_prime[200:300,200:300,200:300].reshape(100**3, 1),
        #                           self.grad_U_bar[200:300,200:300,200:300].reshape(100**3, 1),
        #                           self.grad_V_bar[200:300,200:300,200:300].reshape(100**3, 1),
        #                           self.grad_W_bar[200:300,200:300,200:300].reshape(100**3, 1)
        #                           ])

        filename = join(self.case, 'postProcess_UVW/filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_UVW.pkl')

        dataArray_dd = dd.io.from_dask_array(self.dataArray_da,columns=
                                                  ['c_bar',
                                                   'mag_grad_c_bar',
                                                   'omega_DNS_filtered',
                                                   'omega_model_planar',
                                                   'Xi_iso_085',
                                                   'U_bar',
                                                   'V_bar',
                                                   'W_bar',
                                                   'U_prime',
                                                   'V_prime',
                                                   'W_prime',
                                                   'mag_grad_U',
                                                   'mag_grad_V',
                                                   'mag_grad_W'
                                                   ])

        # # filter the data set and remove unecessary entries
        # dataArray_dd = dataArray_dd[dataArray_dd['c_bar'] > 0.001]
        # dataArray_dd = dataArray_dd[dataArray_dd['c_bar'] < 0.999]
        #
        # dataArray_dd = dataArray_dd[dataArray_dd['grad_U'] != 0.0]

        print('Computing data array ...')
        dataArray_df = dataArray_dd.compute()

        print('Writing output to pickle ...')
        dataArray_df.to_pickle(filename)
        print('Data has been written.\n\n')





################################################
# INCLUDE NOW THE VELOCITY DATA UWV.dat to compute sl
# 20.4.2020

class dns_analysis_tensors(dns_analysis_UVW):
    '''
        Analysis includes now the velocity data
    '''

    def __init__(self,case, eps_factor,c_iso_values):
        # extend super class with additional input parameters
        super(dns_analysis_tensors, self).__init__(case, eps_factor,c_iso_values)

        self.rho_c_filtered = None
        self.rho_filtered = None
        self.c_tilde = None
        self.SGS_scalar_flux = None
        self.UP_delta = None

        print('This is the version that computes gradient tensors on DNS and LES grid ...')

    # constructor end

    def compute_sgs_flux(self):
        '''
        Computes the SGS scalar flux: research_progress_shin_8.pdf: slide 6
        :return:
        '''
        print('computing SGS scalar flux ...')
        # compute the components of u*c
        U_c_bar = self.apply_filter(self.c_data_np * self.U)
        V_c_bar = self.apply_filter(self.c_data_np * self.V)
        W_c_bar = self.apply_filter(self.c_data_np * self.W)

        TAU_11 = U_c_bar - self.c_filtered * self.U_bar
        TAU_22 = V_c_bar - self.c_filtered * self.V_bar
        TAU_33 = W_c_bar - self.c_filtered * self.W_bar

        self.SGS_scalar_flux = np.sqrt(TAU_11 **2 + TAU_22 ** 2 + TAU_33**2)

    def compute_UP_delta(self):
        '''
        #compute U_prime sgs according to Junsu, slide 6, research_progress_shin_8.pptx
        :return:
        '''

        print('computing UP_delta ...')

        U_rho_bar = self.apply_filter(self.U*self.rho_data_np)
        V_rho_bar = self.apply_filter(self.V*self.rho_data_np)
        W_rho_bar = self.apply_filter(self.W*self.rho_data_np)

        UU_rho_bar = self.apply_filter(self.U*self.U*self.rho_data_np)
        VV_rho_bar = self.apply_filter(self.V*self.V*self.rho_data_np)
        WW_rho_bar = self.apply_filter(self.W*self.W*self.rho_data_np)

        TAU11 = UU_rho_bar - (U_rho_bar**2 / self.rho_filtered)
        TAU22 = VV_rho_bar - (V_rho_bar**2 / self.rho_filtered)
        TAU33 = WW_rho_bar - (W_rho_bar**2 / self.rho_filtered)

        self.UP_delta = np.sqrt((TAU11+TAU22+TAU33)/(3.0*self.rho_filtered) )



    def run_analysis_tensors(self,filter_width ,filter_type, c_analytical=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        ##Todo: muss noch überarbeitet werden!

        self.every_nth = 1

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        #('Reading in U fieds ...')
        self.read_UVW()

        # filter the c and rho field
        print('Filtering c, rho*c, rho fields ')
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)
        self.rho_c_filtered = self.apply_filter(self.rho_c_data_np)
        self.rho_filtered = self.apply_filter(self.rho_data_np)

        print('computing c_tilde')
        self.c_tilde = self.rho_c_filtered/ self.rho_filtered

        #Todo: In paralllel? How?
        print('Filtering U components')
        self.U_bar = self.apply_filter(self.U)
        self.V_bar = self.apply_filter(self.V)
        self.W_bar = self.apply_filter(self.W)


        ###########################
        print('Computing U primes')
        # imported cython function: dtype needs to be float32!
        U_prime, V_prime, W_prime = compute_U_prime_cython(np.ascontiguousarray(self.U, dtype=np.float32),
                                                            np.ascontiguousarray(self.V, dtype=np.float32),
                                                            np.ascontiguousarray(self.W, dtype=np.float32),
                                                            self.Nx, self.Ny, self.Nz, self.filter_width)

        #  CONVERT BACK TO NUMPY ARRAY FROM CYTHON
        self.U_prime = np.asarray(U_prime)
        self.V_prime = np.asarray(V_prime)
        self.W_prime = np.asarray(W_prime)



        # free memory and delete U fields
        del self.U, self.W, self.V, U_prime, V_prime, W_prime

        # ###########################
        # compute the magnitude of gradients of the velocity field on the LES mesh
        #self.compute_gradU_LES_4thO()
        # use imported cython function: dtype needs to be float32!
        grad_U_x_LES, grad_V_x_LES, grad_W_x_LES, grad_U_y_LES, grad_V_y_LES, grad_W_y_LES, grad_U_z_LES, grad_V_z_LES, grad_W_z_LES = \
                                                                    compute_gradU_LES_4thO_tensor_cython(
                                                                    np.ascontiguousarray(self.U_bar, dtype=np.float32),
                                                                    np.ascontiguousarray(self.V_bar, dtype=np.float32),
                                                                    np.ascontiguousarray(self.W_bar, dtype=np.float32),
                                                                    self.Nx, self.Ny, self.Nz,
                                                                    np.float32(self.delta_x),int(self.filter_width))

        # #  CONVERT BACK TO NUMPY ARRAY FROM CYTHON
        grad_U_x_LES = np.asarray(grad_U_x_LES)
        grad_V_x_LES = np.asarray(grad_V_x_LES)
        grad_W_x_LES = np.asarray(grad_W_x_LES)
        grad_U_y_LES = np.asarray(grad_U_y_LES)
        grad_V_y_LES = np.asarray(grad_V_y_LES)
        grad_W_y_LES = np.asarray(grad_W_y_LES)
        grad_U_z_LES = np.asarray(grad_U_z_LES)
        grad_V_z_LES = np.asarray(grad_V_z_LES)
        grad_W_z_LES = np.asarray(grad_W_z_LES)


        # compute the U gradients on DNS mesh
        grad_U_x_DNS, grad_V_x_DNS, grad_W_x_DNS, grad_U_y_DNS, grad_V_y_DNS, grad_W_y_DNS, grad_U_z_DNS, grad_V_z_DNS, grad_W_z_DNS = \
                                                                    compute_gradU_LES_4thO_tensor_cython(
                                                                    np.ascontiguousarray(self.U_bar, dtype=np.float32),
                                                                    np.ascontiguousarray(self.V_bar, dtype=np.float32),
                                                                    np.ascontiguousarray(self.W_bar, dtype=np.float32),
                                                                    self.Nx, self.Ny, self.Nz,
                                                                    np.float32(self.delta_x),int(1))

        # #  CONVERT BACK TO NUMPY ARRAY FROM CYTHON
        grad_U_x_DNS = np.asarray(grad_U_x_DNS)
        grad_V_x_DNS = np.asarray(grad_V_x_DNS)
        grad_W_x_DNS = np.asarray(grad_W_x_DNS)
        grad_U_y_DNS = np.asarray(grad_U_y_DNS)
        grad_V_y_DNS = np.asarray(grad_V_y_DNS)
        grad_W_y_DNS = np.asarray(grad_W_y_DNS)
        grad_U_z_DNS = np.asarray(grad_U_z_DNS)
        grad_V_z_DNS = np.asarray(grad_V_z_DNS)
        grad_W_z_DNS = np.asarray(grad_W_z_DNS)

        ###########################
        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)


        ###########################
        # compute abs(grad(xi)) on the whole DNS domain
        self.convert_c_field_to_xi()
        # self.grad_xi_DNS = self.compute_DNS_grad_xi_4thO()
        # CYTHON
        print('Computing gradients of xi on DNS mesh 4th Order with Cython')
        grad_xi_DNS = compute_DNS_grad_4thO_cython(np.ascontiguousarray(self.xi_np, dtype=np.float32),
                                                      self.Nx, self.Ny, self.Nz, np.float32(self.delta_x))

        self.grad_xi_DNS = np.asarray(grad_xi_DNS)

        del grad_xi_DNS     # free memory


        ###########################
        # compute abs(grad(xi)) on the whole DNS domain
        # CYTHON
        print('Computing x,y,z gradients of c_bar on DNS mesh 4th Order with Cython')
        grad_c_x_DNS, grad_c_y_DNS, grad_c_z_DNS = compute_LES_grad_4thO_tensor_cython(np.ascontiguousarray(self.c_filtered, dtype=np.float32),
                                                      self.Nx, self.Ny, self.Nz, np.float32(self.delta_x), int(1))

        grad_c_x_DNS = np.asarray(grad_c_x_DNS)
        grad_c_y_DNS = np.asarray(grad_c_y_DNS)
        grad_c_z_DNS = np.asarray(grad_c_z_DNS)


        print('Computing x,y,z gradients of c_bar on LES mesh 4th Order with Cython')
        grad_c_x_LES, grad_c_y_LES, grad_c_z_LES = compute_LES_grad_4thO_tensor_cython(np.ascontiguousarray(self.c_filtered, dtype=np.float32),
                                                      self.Nx, self.Ny, self.Nz, np.float32(self.delta_x), int(self.filter_width))

        grad_c_x_LES = np.asarray(grad_c_x_LES)
        grad_c_y_LES = np.asarray(grad_c_y_LES)
        grad_c_z_LES = np.asarray(grad_c_z_LES)

        #del grad_xi_DNS     # free memory

        ###########################
        # compute omega based on pfitzner
        self.compute_Pfitzner_model()

        # 2nd method to compute Xi
        Xi_iso_085, dirac_grad_xi_arr = self.compute_Xi_iso_dirac_xi(c_iso=0.85)

        # #creat dask array and reshape all data

        self.dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx * self.Ny * self.Nz,1),

                                          self.omega_DNS_filtered.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.omega_model_cbar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          Xi_iso_085.reshape(self.Nx * self.Ny * self.Nz,1),
                                          self.U_bar.reshape(self.Nx * self.Ny * self.Nz,1),
                                          self.V_bar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.W_bar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.U_prime.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.V_prime.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.W_prime.reshape(self.Nx * self.Ny * self.Nz, 1),

                                  ])

        self.dataArray_LES_grads_da = da.hstack([
                                        #self.c_filtered.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_c_x_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_c_y_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_c_z_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_c_x_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_c_y_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_c_z_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),

                                       grad_U_x_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_V_x_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_W_x_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_U_y_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_V_y_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_W_y_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_U_z_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_V_z_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_W_z_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_U_x_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_V_x_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_W_x_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_U_y_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_V_y_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_W_y_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_U_z_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_V_z_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_W_z_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       ])

        self.dataArray_DNS_grads_da = da.hstack([

                                       # grad_c_x_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_c_y_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_c_z_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_c_x_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_c_y_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_c_z_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),

                                       # grad_U_x_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_V_x_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_W_x_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_U_y_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_V_y_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_W_y_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_U_z_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_V_z_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       # grad_W_z_LES.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_U_x_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_V_x_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_W_x_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_U_y_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_V_y_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_W_y_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_U_z_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_V_z_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       grad_W_z_DNS.reshape(self.Nx * self.Ny * self.Nz, 1),
                                       ])

        filename = join(self.case, 'postProcess_UVW/filter_width_' + self.filter_type + '_' + str(self.filter_width) + '_tensor.pkl')
        filename_LES = join(self.case, 'postProcess_UVW/filter_width_' + self.filter_type + '_' + str(
            self.filter_width) + '_grad_LES_tensor.pkl')
        filename_DNS = join(self.case, 'postProcess_UVW/filter_width_' + self.filter_type + '_' + str(
            self.filter_width) + '_grad_DNS_tensor.pkl')

        dataArray_dd = dd.io.from_dask_array(self.dataArray_da,columns=
                                                  ['c_bar',
                                                   # 'grad_c_x_LES',
                                                   # 'grad_c_y_LES',
                                                   # 'grad_c_z_LES',
                                                   # 'grad_c_x_DNS',
                                                   # 'grad_c_y_DNS',
                                                   # 'grad_c_z_DNS',
                                                   'omega_DNS_filtered',
                                                   'omega_model_planar',
                                                   'Xi_iso_085',
                                                   'U_bar',
                                                   'V_bar',
                                                   'W_bar',
                                                   'U_prime',
                                                   'V_prime',
                                                   'W_prime',
                                                   # 'grad_U_x_LES',
                                                   # 'grad_V_x_LES',
                                                   # 'grad_W_x_LES',
                                                   # 'grad_U_y_LES',
                                                   # 'grad_V_y_LES',
                                                   # 'grad_W_y_LES',
                                                   # 'grad_U_z_LES',
                                                   # 'grad_V_z_LES',
                                                   # 'grad_W_z_LES',
                                                   # 'grad_U_x_DNS',
                                                   # 'grad_V_x_DNS',
                                                   # 'grad_W_x_DNS',
                                                   # 'grad_U_y_DNS',
                                                   # 'grad_V_y_DNS',
                                                   # 'grad_W_y_DNS',
                                                   # 'grad_U_z_DNS',
                                                   # 'grad_V_z_DNS',
                                                   # 'grad_W_z_DNS',
                                                   ])


        dataArray_LES_dd = dd.io.from_dask_array(self.dataArray_LES_grads_da,columns=
                                                  [#'c_bar',
                                                   'grad_c_x_LES',
                                                   'grad_c_y_LES',
                                                   'grad_c_z_LES',
                                                   # 'grad_c_x_DNS',
                                                   # 'grad_c_y_DNS',
                                                   # 'grad_c_z_DNS',
                                                   # 'omega_DNS_filtered',
                                                   # 'omega_model_planar',
                                                   # 'Xi_iso_085',
                                                   # 'U_bar',
                                                   # 'V_bar',
                                                   # 'W_bar',
                                                   # 'U_prime',
                                                   # 'V_prime',
                                                   # 'W_prime',
                                                   'grad_U_x_LES',
                                                   'grad_V_x_LES',
                                                   'grad_W_x_LES',
                                                   'grad_U_y_LES',
                                                   'grad_V_y_LES',
                                                   'grad_W_y_LES',
                                                   'grad_U_z_LES',
                                                   'grad_V_z_LES',
                                                   'grad_W_z_LES',
                                                   # 'grad_U_x_DNS',
                                                   # 'grad_V_x_DNS',
                                                   # 'grad_W_x_DNS',
                                                   # 'grad_U_y_DNS',
                                                   # 'grad_V_y_DNS',
                                                   # 'grad_W_y_DNS',
                                                   # 'grad_U_z_DNS',
                                                   # 'grad_V_z_DNS',
                                                   # 'grad_W_z_DNS',
                                                   ])

        dataArray_DNS_dd = dd.io.from_dask_array(self.dataArray_DNS_grads_da,columns=
                                                  [#'c_bar',
                                                   # 'grad_c_x_LES',
                                                   # 'grad_c_y_LES',
                                                   # 'grad_c_z_LES',
                                                   'grad_c_x_DNS',
                                                   'grad_c_y_DNS',
                                                   'grad_c_z_DNS',
                                                   # 'omega_DNS_filtered',
                                                   # 'omega_model_planar',
                                                   # 'Xi_iso_085',
                                                   # 'U_bar',
                                                   # 'V_bar',
                                                   # 'W_bar',
                                                   # 'U_prime',
                                                   # 'V_prime',
                                                   # 'W_prime',
                                                   # 'grad_U_x_LES',
                                                   # 'grad_V_x_LES',
                                                   # 'grad_W_x_LES',
                                                   # 'grad_U_y_LES',
                                                   # 'grad_V_y_LES',
                                                   # 'grad_W_y_LES',
                                                   # 'grad_U_z_LES',
                                                   # 'grad_V_z_LES',
                                                   # 'grad_W_z_LES',
                                                   'grad_U_x_DNS',
                                                   'grad_V_x_DNS',
                                                   'grad_W_x_DNS',
                                                   'grad_U_y_DNS',
                                                   'grad_V_y_DNS',
                                                   'grad_W_y_DNS',
                                                   'grad_U_z_DNS',
                                                   'grad_V_z_DNS',
                                                   'grad_W_z_DNS',
                                                   ])

        # # filter the data set and remove unecessary entries
        # dataArray_dd = dataArray_dd[dataArray_dd['c_bar'] > 0.001]
        # dataArray_dd = dataArray_dd[dataArray_dd['c_bar'] < 0.999]
        #
        # dataArray_dd = dataArray_dd[dataArray_dd['grad_U'] != 0.0]

        print('Computing data array ...')
        dataArray_df = dataArray_dd.compute()
        dataArray_LES_df = dataArray_LES_dd.compute()
        dataArray_DNS_df = dataArray_DNS_dd.compute()

        print('Writing output to pickle ...')
        dataArray_df.to_pickle(filename)
        print('Writing LES grads to pickle ...')
        dataArray_LES_df.to_pickle(filename_LES)
        print('Writing DNS grads to pickle ...')
        dataArray_DNS_df.to_pickle(filename_DNS)

        print('Data has been written.\n\n')


    # for junsu: 18.5.20
    def run_analysis_tilde(self,filter_width ,filter_type, c_analytical=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        ##Todo: muss noch überarbeitet werden!

        self.every_nth = 1

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        #('Reading in U fieds ...')
        self.read_UVW()

        # filter the c and rho field
        print('Filtering c, rho*c, rho fields ')
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)
        self.rho_c_filtered = self.apply_filter(self.rho_c_data_np)
        self.rho_filtered = self.apply_filter(self.rho_data_np)

        print('computing c_tilde')
        self.c_tilde = self.rho_c_filtered/ self.rho_filtered

        #Todo: In paralllel? How?
        print('Filtering U components')
        self.U_bar = self.apply_filter(self.U)
        self.V_bar = self.apply_filter(self.V)
        self.W_bar = self.apply_filter(self.W)


        # Computing SGS scalar flux
        self.compute_sgs_flux()

        # Computing UP_delta
        self.compute_UP_delta()


        ###########################
        # U PRIMES
        #check if U_prime has already been computed!
        Uprime_filename = 'filter_width_TOPHAT_%i_tensor.pkl' % self.filter_width

        file_list = os.listdir(join(self.case,'postProcess_UVW'))

        if Uprime_filename in file_list: #os.path.isdir(Uprime_filename):
            print('Using U_primes form: ',Uprime_filename)

            data_df = pd.read_pickle(join(self.case,'postProcess_UVW',Uprime_filename))

            self.U_prime = data_df['U_prime'].values.reshape(512, 512, 512)
            self.V_prime = data_df['V_prime'].values.reshape(512, 512, 512)
            self.W_prime = data_df['W_prime'].values.reshape(512, 512, 512)

            del data_df

        else:
            print('Computing U primes')
            # imported cython function: dtype needs to be float32!
            U_prime, V_prime, W_prime = compute_U_prime_cython(np.ascontiguousarray(self.U, dtype=np.float32),
                                                                np.ascontiguousarray(self.V, dtype=np.float32),
                                                                np.ascontiguousarray(self.W, dtype=np.float32),
                                                                self.Nx, self.Ny, self.Nz, self.filter_width)

            #  CONVERT BACK TO NUMPY ARRAY FROM CYTHON
            self.U_prime = np.asarray(U_prime)
            self.V_prime = np.asarray(V_prime)
            self.W_prime = np.asarray(W_prime)

            # free memory and delete U fields
            del self.U, self.W, self.V, U_prime, V_prime, W_prime


        ###########################
        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)


        ###########################
        # compute abs(grad(xi)) on the whole DNS domain
        self.convert_c_field_to_xi()
        # self.grad_xi_DNS = self.compute_DNS_grad_xi_4thO()
        # CYTHON
        print('Computing gradients of xi on DNS mesh 4th Order with Cython')
        grad_xi_DNS = compute_DNS_grad_4thO_cython(np.ascontiguousarray(self.xi_np, dtype=np.float32),
                                                      self.Nx, self.Ny, self.Nz, np.float32(self.delta_x))

        self.grad_xi_DNS = np.asarray(grad_xi_DNS)

        del grad_xi_DNS     # free memory


        ###########################
        # compute omega based on pfitzner
        self.compute_Pfitzner_model()

        # 2nd method to compute Xi
        Xi_iso_085, dirac_grad_xi_arr = self.compute_Xi_iso_dirac_xi(c_iso=0.85)

        # #creat dask array and reshape all data

        self.dataArray_da = da.hstack([self.c_filtered.reshape(self.Nx * self.Ny * self.Nz,1),
                                          self.c_tilde.reshape(self.Nx * self.Ny * self.Nz,1),
                                          self.omega_DNS_filtered.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.omega_model_cbar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          Xi_iso_085.reshape(self.Nx * self.Ny * self.Nz,1),
                                          self.U_bar.reshape(self.Nx * self.Ny * self.Nz,1),
                                          self.V_bar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.W_bar.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.U_prime.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.V_prime.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.W_prime.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.SGS_scalar_flux.reshape(self.Nx * self.Ny * self.Nz, 1),
                                          self.UP_delta.reshape(self.Nx * self.Ny * self.Nz, 1),
                                    ])

        filename = join(self.case, 'postProcess_UVW/filter_width_' + str(self.filter_width) + '_tilde.pkl')

        dataArray_dd = dd.io.from_dask_array(self.dataArray_da,columns=
                                                  ['c_bar',
                                                    'c_tilde',
                                                   'omega_DNS_filtered',
                                                   'omega_model_planar',
                                                   'Xi_iso_085',
                                                   'U_bar',
                                                   'V_bar',
                                                   'W_bar',
                                                   'U_prime',
                                                   'V_prime',
                                                   'W_prime',
                                                   'scalar_flux_SGS',
                                                   'UP_delta',
                                                   ])


        # # filter the data set and remove unecessary entries
        # dataArray_dd = dataArray_dd[dataArray_dd['c_bar'] > 0.001]
        # dataArray_dd = dataArray_dd[dataArray_dd['c_bar'] < 0.999]
        #
        # dataArray_dd = dataArray_dd[dataArray_dd['grad_U'] != 0.0]

        print('Computing data array ...')
        dataArray_df = dataArray_dd.compute()


        print('Writing output to pickle ...')
        dataArray_df.to_pickle(filename)


        print('Data has been written.\n\n')

    def run_analysis_UVW(self,filter_width ,filter_type, c_analytical=False):
        return NotImplementedError



################################################
# Prepare data sets for DNN regression of omega_filtered
# 8.9.2020

class dns_analysis_prepareDNN(dns_analysis_dirac_FSD_alt):
    '''
        Analysis includes now the velocity data
    '''

    def __init__(self,case,data_format ):
        # extend super class with additional input parameters

        # hard coded default values
        eps_factor=1
        c_iso_values =[0.85]

        super(dns_analysis_prepareDNN, self).__init__(case, eps_factor,c_iso_values)

        print('This is the version that prepares the data for DNN regression ...')

        self.data_format = data_format
        print('**********')
        print('wirte data to format: ',data_format)
        print('**********')

        self.U = None
        self.V = None
        self.W = None

        self.U_bar = None
        self.V_bar = None
        self.W_bar = None

        self.U_prime = None
        self.V_prime = None
        self.W_prime = None

        # create empty array
        self.grad_U_bar = np.zeros([self.Nx, self.Ny, self.Nz])
        self.grad_V_bar = np.zeros([self.Nx, self.Ny, self.Nz])
        self.grad_W_bar = np.zeros([self.Nx, self.Ny, self.Nz])

    # constructor end


    def read_UVW(self):
        '''
        reads in the velocity fields and stores it as np.array. 3D!!
        :return: nothing
        '''

        print('Reading the velocity data ...')

        U_np = pd.read_csv(join(self.case, 'U.dat'), names=['U']).values.astype(dtype=np.float32)
        V_np = pd.read_csv(join(self.case, 'V.dat'), names=['U']).values.astype(dtype=np.float32)
        W_np = pd.read_csv(join(self.case, 'W.dat'), names=['U']).values.astype(dtype=np.float32)

        self.U = U_np.reshape(self.Nx,self.Ny,self.Nz)
        self.V = V_np.reshape(self.Nx,self.Ny,self.Nz)
        self.W = W_np.reshape(self.Nx,self.Ny,self.Nz)

        # delete to save memory
        del U_np, V_np, W_np

    #@jit(nopython=True, parallel=True)
    def compute_U_prime(self):
        '''
        # compute the SGS velocity fluctuations
        # u_prime is the STD of the U-DNS components within a LES cell
        :return: nothing
        '''
        print('Computing U prime')
        self.U_prime = scipy.ndimage.generic_filter(self.U, np.var, mode='wrap',
                                                    size=(self.filter_width, self.filter_width,self.filter_width))
        print('Computing V prime')
        self.V_prime = scipy.ndimage.generic_filter(self.V, np.var, mode='wrap',
                                                    size=(self.filter_width, self.filter_width, self.filter_width))
        print('Computing W prime')
        self.W_prime = scipy.ndimage.generic_filter(self.W, np.var, mode='wrap',
                                                    size=(self.filter_width, self.filter_width, self.filter_width))


    def compute_sgs_flux(self):
        '''
        Computes the SGS scalar flux: research_progress_shin_8.pdf: slide 6
        :return:
        '''
        print('computing SGS scalar flux ...')
        # compute the components of u*c
        U_c_bar = self.apply_filter(self.c_data_np * self.U)
        V_c_bar = self.apply_filter(self.c_data_np * self.V)
        W_c_bar = self.apply_filter(self.c_data_np * self.W)

        TAU_11 = U_c_bar - self.c_filtered * self.U_bar
        TAU_22 = V_c_bar - self.c_filtered * self.V_bar
        TAU_33 = W_c_bar - self.c_filtered * self.W_bar

        self.SGS_scalar_flux = np.sqrt(TAU_11 **2 + TAU_22 ** 2 + TAU_33**2)

    def compute_UP_delta(self):
        '''
        #compute U_prime sgs according to Junsu, slide 6, research_progress_shin_8.pptx
        :return:
        '''

        print('computing UP_delta ...')

        U_rho_bar = self.apply_filter(self.U*self.rho_data_np)
        V_rho_bar = self.apply_filter(self.V*self.rho_data_np)
        W_rho_bar = self.apply_filter(self.W*self.rho_data_np)

        UU_rho_bar = self.apply_filter(self.U*self.U*self.rho_data_np)
        VV_rho_bar = self.apply_filter(self.V*self.V*self.rho_data_np)
        WW_rho_bar = self.apply_filter(self.W*self.W*self.rho_data_np)

        TAU11 = UU_rho_bar - (U_rho_bar**2 / self.rho_filtered)
        TAU22 = VV_rho_bar - (V_rho_bar**2 / self.rho_filtered)
        TAU33 = WW_rho_bar - (W_rho_bar**2 / self.rho_filtered)

        self.UP_delta = np.sqrt((TAU11+TAU22+TAU33)/(3.0*self.rho_filtered) )

    #@jit(nopython=True, parallel=True)
    def compute_gradU_LES_4thO(self):
        # '''
        # Compute the magnitude of the gradient of the DNS c-field, based on neighbour cells
        # 4th Order central differencing
        # :return: nothing
        # '''

        print('Computing gradients of U_bar on DNS mesh 4th Order...')

        # compute gradients from the boundaries away ...
        for l in range(2,self.Nx-2):
            for m in range(2,self.Ny-2):
                for n in range(2,self.Nz-2):
                    this_U_gradX = (-self.U_bar[l+2, m, n] + 8*self.U_bar[l+1,m, n] - 8*self.U_bar[l-1,m, n] + self.U_bar[l-2, m, n])/(12 * self.delta_x)
                    this_U_gradY = (-self.U_bar[l, m+2, n] + 8*self.U_bar[l,m+1, n] - 8*self.U_bar[l,m-1, n] + self.U_bar[l, m-2, n])/(12 * self.delta_x)
                    this_U_gradZ = (-self.U_bar[l, m, n+2] + 8*self.U_bar[l,m, n+1] - 8*self.U_bar[l,m, n-1] + self.U_bar[l, m, n+2])/(12 * self.delta_x)

                    this_V_gradX = (-self.V_bar[l+2, m, n] + 8*self.V_bar[l+1,m, n] - 8*self.V_bar[l-1,m, n] + self.V_bar[l-2, m, n])/(12 * self.delta_x)
                    this_V_gradY = (-self.V_bar[l, m+2, n] + 8*self.V_bar[l,m+1, n] - 8*self.V_bar[l,m-1, n] + self.V_bar[l, m-2, n])/(12 * self.delta_x)
                    this_V_gradZ = (-self.V_bar[l, m, n+2] + 8*self.V_bar[l,m, n+1] - 8*self.V_bar[l,m, n-1] + self.V_bar[l, m, n+2])/(12 * self.delta_x)

                    this_W_gradX = (-self.W_bar[l+2, m, n] + 8*self.W_bar[l+1,m, n] - 8*self.W_bar[l-1,m, n] + self.W_bar[l-2, m, n])/(12 * self.delta_x)
                    this_W_gradY = (-self.W_bar[l, m+2, n] + 8*self.W_bar[l,m+1, n] - 8*self.W_bar[l,m-1, n] + self.W_bar[l, m-2, n])/(12 * self.delta_x)
                    this_W_gradZ = (-self.W_bar[l, m, n+2] + 8*self.W_bar[l,m, n+1] - 8*self.W_bar[l,m, n-1] + self.W_bar[l, m, n+2])/(12 * self.delta_x)

                    # compute the magnitude of the gradient
                    this_magGrad_U = np.sqrt(this_U_gradX ** 2 + this_U_gradY ** 2 + this_U_gradZ ** 2)
                    this_magGrad_V = np.sqrt(this_V_gradX ** 2 + this_V_gradY ** 2 + this_V_gradZ ** 2)
                    this_magGrad_W = np.sqrt(this_W_gradX ** 2 + this_W_gradY ** 2 + this_W_gradZ ** 2)

                    self.grad_U_bar[l, m, n] = this_magGrad_U
                    self.grad_V_bar[l, m, n] = this_magGrad_V
                    self.grad_W_bar[l, m, n] = this_magGrad_W

    def crop_reshape_dataset(self,data):
        '''
        Crop the boundaries from the cube: Nx-filter_width at each side. Reshape it to a vector
        :param data:
        :return:
        '''

        assert data.shape[0] == self.Nx

        new_length = self.Nx - 2*self.filter_width
        # crop the data
        data = data[self.filter_width:self.Nx-self.filter_width, self.filter_width:self.Nx-self.filter_width, self.filter_width:self.Nx-self.filter_width]

        # reshape  cube to vector
        data = data.reshape(new_length**3,1)

        return data

    def crop_split_dataset(self,data):
        '''
        split in train test data set consistently for all data sets!
        Crop the boundaries from the cube: Nx-filter_width at each side. Reshape it to a vector
        :param data:
        :return:
        '''

        test_range=range(236,276)   # indices of data locations to extract from the cube
        len_test_range = len(test_range)

        assert data.shape[0] == self.Nx

        data_test = data[test_range,test_range,test_range]

        data_train = data

        # set the training data to zero in the location where test data is extracted
        data_train[test_range,test_range,test_range] = 0

        new_length = self.Nx - 2*self.filter_width
        # crop the data
        data_train = data_train[self.filter_width:self.Nx-self.filter_width, self.filter_width:self.Nx-self.filter_width, self.filter_width:self.Nx-self.filter_width]

        # reshape  cube to vector
        data_train = data_train.reshape(new_length**3,1)
        data_test = data_test.reshape(len_test_range**3,1)

        return data_train, data_test


    def run_analysis_DNN(self,filter_width ,filter_type, c_analytical=False):
        '''
        :param filter_width: DNS points to filter
        :param filter_type: use 'TOPHAT' rather than 'GAUSSIAN
        :param c_analytical: compute c minus analytically
        :return:
        '''
        # run the analysis and compute the wrinkling factor -> real 3D cases
        # interval is like nth point, skips some nodes
        self.filter_type = filter_type

        ##Todo: muss noch überarbeitet werden!

        self.every_nth = 1

        print('You are using %s filter!' % self.filter_type)

        self.filter_width = int(filter_width)

        self.c_analytical = c_analytical
        if self.c_analytical is True:
            print('You are using Hypergeometric function for c_minus (Eq.35)!')

        #('Reading in U fieds ...')
        self.read_UVW()

        # filter the c and rho field
        print('Filtering c field ')
        #self.rho_filtered = self.apply_filter(self.rho_data_np)
        self.c_filtered = self.apply_filter(self.c_data_np)
        self.rho_filtered = self.apply_filter(self.rho_data_np)

        #Todo: In paralllel? How?
        print('Filtering U components')
        self.U_bar = self.apply_filter(self.U)
        self.V_bar = self.apply_filter(self.V)
        self.W_bar = self.apply_filter(self.W)

        #############################
        # Compute gradients of c_bar with Cython
        # print('Computing x,y,z gradients of c_bar on LES mesh 4th Order with Cython')
        # grad_c_x_LES, grad_c_y_LES, grad_c_z_LES = compute_LES_grad_4thO_tensor_cython(np.ascontiguousarray(self.c_filtered, dtype=np.float32),
        #                                               self.Nx, self.Ny, self.Nz, np.float32(self.delta_x), int(self.filter_width))

        # grad_c_x_LES = np.asarray(grad_c_x_LES)
        # grad_c_y_LES = np.asarray(grad_c_y_LES)
        # grad_c_z_LES = np.asarray(grad_c_z_LES)

        print('Computing x,y,z gradients of c_bar on LES mesh')
        grad_c_x_LES, grad_c_y_LES, grad_c_z_LES = np.gradient(self.c_filtered,int(self.filter_width))

        print('Compute UP_delta')
        self.compute_UP_delta()

        print('Compute Scalar flux')
        self.compute_sgs_flux()

        # ###########################
        # compute the magnitude of gradients of the velocity field on the LES mesh
        #self.compute_gradU_LES_4thO()
        # use imported cython function: dtype needs to be float32!
        # grad_U_x_LES, grad_V_x_LES, grad_W_x_LES, grad_U_y_LES, grad_V_y_LES, grad_W_y_LES, grad_U_z_LES, grad_V_z_LES, grad_W_z_LES = \
        #                                                             compute_gradU_LES_4thO_tensor_cython(
        #                                                             np.ascontiguousarray(self.U_bar, dtype=np.float32),
        #                                                             np.ascontiguousarray(self.V_bar, dtype=np.float32),
        #                                                             np.ascontiguousarray(self.W_bar, dtype=np.float32),
        #                                                             self.Nx, self.Ny, self.Nz,
        #                                                             np.float32(self.delta_x),int(self.filter_width))

        # # #  CONVERT BACK TO NUMPY ARRAY FROM CYTHON
        # grad_U_x_LES = np.asarray(grad_U_x_LES)
        # grad_V_x_LES = np.asarray(grad_V_x_LES)
        # grad_W_x_LES = np.asarray(grad_W_x_LES)
        # grad_U_y_LES = np.asarray(grad_U_y_LES)
        # grad_V_y_LES = np.asarray(grad_V_y_LES)
        # grad_W_y_LES = np.asarray(grad_W_y_LES)
        # grad_U_z_LES = np.asarray(grad_U_z_LES)
        # grad_V_z_LES = np.asarray(grad_V_z_LES)
        # grad_W_z_LES = np.asarray(grad_W_z_LES)

        # #  CONVERT BACK TO NUMPY ARRAY FROM CYTHON
        print('Computing x,y,z gradients of Velocity on LES mesh')
        grad_U_x_LES, grad_U_y_LES, grad_U_z_LES = np.gradient(self.U_bar,int(self.filter_width))
        grad_V_x_LES, grad_V_y_LES, grad_V_z_LES = np.gradient(self.V_bar,int(self.filter_width))
        grad_W_x_LES, grad_W_y_LES, grad_W_z_LES = np.gradient(self.W_bar,int(self.filter_width))

        ###########################
        # Compute the scaled Delta (Pfitzner PDF)
        self.Delta_LES = self.delta_x*self.filter_width * self.Sc * self.Re * np.sqrt(self.p/self.p_0)
        print('Delta_LES is: %.3f' % self.Delta_LES)
        flame_thickness = self.compute_flamethickness()
        print('Flame thickness: ',flame_thickness)

        #maximum possible wrinkling factor
        print('Maximum possible wrinkling factor: ', self.Delta_LES/flame_thickness)



        ###########################
        # compute omega based on pfitzner
        self.compute_Pfitzner_model()

        # # 2nd method to compute Xi
        # Xi_iso_085, dirac_grad_xi_arr = self.compute_Xi_iso_dirac_xi(c_iso=0.85)

        # #creat dask array and reshape all data

        output_list =[self.c_filtered,
                                  self.omega_DNS_filtered,
                                  self.omega_model_cbar,
                                  self.U_bar,
                                  self.V_bar,
                                  self.W_bar,
                                  grad_c_x_LES,
                                  grad_c_y_LES,
                                  grad_c_z_LES,
                                    grad_U_x_LES,
                                    grad_V_x_LES,
                                    grad_W_x_LES,
                                    grad_U_y_LES,
                                    grad_V_y_LES,
                                    grad_W_y_LES,
                                    grad_U_z_LES,
                                    grad_V_z_LES,
                                    grad_W_z_LES,
                                    self.UP_delta,
                                    self.SGS_scalar_flux,
                                    # add the filter width as information and perturb it slightly
                                    np.ones((self.Nx,self.Ny,self.Nz))*self.Delta_LES +
                                        (np.random.rand(self.Nx,self.Ny,self.Nz)*1e-6)
                                  ]

        output_names =['c_bar',
                         # 'mag_grad_c_bar',
                         'omega_DNS_filtered',
                         'omega_model_planar',
                         'U_bar',
                         'V_bar',
                         'W_bar',
                         'grad_c_x_LES',
                         'grad_c_y_LES',
                         'grad_c_z_LES',
                         'grad_U_x_LES',
                         'grad_V_x_LES',
                         'grad_W_x_LES',
                         'grad_U_y_LES',
                         'grad_V_y_LES',
                         'grad_W_y_LES',
                         'grad_U_z_LES',
                         'grad_V_z_LES',
                         'grad_W_z_LES',
                         'UP_delta',
                         'SGS_flux',
                         'Delta_LES'
                         ]

        # reshape and crop the list that is to be written to file
        output_test = []
        output_train = []

        for x in output_list:
            this_train, this_test = self.crop_split_dataset(x)

            output_train.append(this_train)
            output_test.append(this_test)


        #output_list = [self.crop_reshape_dataset(x) for x in output_list]

        dataArray_train_da = da.hstack(output_train)
        dataArray_test_da = da.hstack(output_test)


        filename = join(self.case, 'postProcess_DNN/filter_width_'  + str(self.filter_width) + '_DNN')

        dataArray_train_dd = dd.io.from_dask_array(dataArray_train_da,columns=output_names)
        dataArray_test_dd = dd.io.from_dask_array(dataArray_test_da, columns=output_names)

        # filter the data set and remove unecessary entries
        print('filtering for c<0.99 & c>0.01 ...')
        dataArray_train_dd = dataArray_train_dd[(dataArray_train_dd['c_bar'] > 0.01) &(dataArray_train_dd['c_bar'] < 0.99)]
        dataArray_test_dd = dataArray_test_dd[(dataArray_test_dd['c_bar'] > 0.01) &(dataArray_test_dd['c_bar'] < 0.99)]


        if self.data_format == 'hdf':
            print('Writing output to hdf ...')
            dataArray_df = dataArray_train_dd.compute()
            dataArray_df.to_hdf(filename + '_train.hdf',key='DNS',format='table')
            dataArray_df = dataArray_test_dd.compute()
            dataArray_df.to_hdf(filename + '_test.hdf', key='DNS', format='table')

        elif self.data_format == 'parquet':
            print('Writing output to parquet ...')
            dataArray_df = dataArray_train_dd.compute()
            dataArray_df.to_parquet(filename + '_train.parquet')
            dataArray_df = dataArray_test_dd.compute()
            dataArray_df.to_parquet(filename + '_test.parquet')

        elif self.data_format == 'csv':
            print('Writing output to csv ..')
            dataArray_df = dataArray_train_dd.compute()
            dataArray_df.to_csv(filename + '_train.csv')
            dataArray_df = dataArray_test_dd.compute()
            dataArray_df.to_csv(filename + '_test.csv')

        print('Data has been written.\n\n')


