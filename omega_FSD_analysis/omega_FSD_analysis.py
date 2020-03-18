# read in the iso slices and the omega_DNS 3D fields
# 11.3.20

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import dask.dataframe as dd
#from numba import jit
#from mayavi import mlab
# to free memory
import scipy.ndimage
import scipy as sc
import dask.array as da
import sys
from scipy import special, interpolate
import time
from scipy.integrate import simps,trapz, cumtrapz, dblquad, newton_cotes, quadrature

m = 4.4545
beta = 6
alpha = 0.81818
eps_factor=100
delta_x = 1/220

c_iso_values=[0.001, 0.3,0.5,0.55,0.6,0.65,0.7,0.725,0.75,0.775,0.8,0.82,0.84,0.85,0.86,0.88,0.9,0.92,0.94,0.98 ,0.999]
c_profile_1D = pd.read_csv('1D_data_cube.csv')['c']

# read in stuff
# this is the dirac*(m+1)c_iso**m
iso_dirac_reaction_rates = np.load('filter_width_32_iso_fields.npy')

omega_DNS = np.load('filter_width_32_omega_DNS.npy')

#%%
def analytical_omega(c):

    exponent = - (beta * (1 - c)) / (1 - alpha * (1 - c))
    Eigenval = 18.97  # beta**2 / 2 + beta*(3*alpha - 1.344)

    om_analytical = Eigenval * ((1 - alpha * (1 - c))) ** (-1) * (1 - c) * np.exp(exponent)

    return om_analytical


def omega_over_dc_dxi(c):
    ''' omega_m/ (dc/dxi) = (m+1)*c**m '''
    return (m+1)*c**m


def convert_to_xi(c):
    # converts the c-field to the xi field (Eq. 13, Pfitzner)
    c_clipped = c*0.9999 + 1e-5

    xi_np = 1/m * np.log(c_clipped**m/ (1 - c_clipped**m) )
    #xi_iso_values = [1/m * np.log(c**m/ (1 - c**m) ) for c in c_iso_values]
    return xi_np


def compute_dirac_cos(c_iso,eps_factor):
    '''
    EVERYTHING IS COMPUTED IN xi SPACE!
    :param xi_phi: xi(x) - xi_iso
    :param m: scaling factor (Zahedi paper)
    :return: numerical dirac delta function for whole field
    '''
    eps = eps_factor*delta_x

    xi_np = convert_to_xi(c_profile_1D)
    xi_iso= convert_to_xi(c_iso)
    xi_phi=abs(xi_np - xi_iso)

    X = xi_phi/eps

    dirac_vec = np.zeros(len(X))

    # Fallunterscheidung für X < 0
    for id, x in enumerate(X):
        if x < 1:
            dirac_vec[id] =1/(2*eps) * (1 + np.cos(np.pi * x)) * delta_x
            #print('dirac_vec: ',dirac_vec[id])

    return dirac_vec

#%%

n_points = 32
Delta_LES = delta_x*n_points * 0.7 * 50 * np.sqrt(1/1)

omega_iso_array = np.zeros([len(c_iso_values),len(c_profile_1D)])
for id,c_iso in enumerate(c_iso_values):
    dirac = compute_dirac_cos(c_iso=c_iso,eps_factor=100)
    omega_iso_array[id,:] = dirac * omega_over_dc_dxi(c_iso)
    plt.plot(omega_iso_array[id,:],'-')

plt.show()

omega_integrated = np.sum(omega_iso_array, axis=0) #simps(omega_iso_array,c_iso_values,axis=0)
omega_analytical = analytical_omega(c_profile_1D)

omega_integrated_filtered = np.convolve(omega_integrated,np.ones(n_points,dtype=int),'same')/Delta_LES
#sc.ndimage.filters.uniform_filter(omega_integrated,size=int(n_points), mode='reflect')
omega_analytical_filtered = sc.ndimage.filters.uniform_filter(omega_analytical,size=int(n_points),mode='reflect')

plt.plot(omega_integrated_filtered[220:300] ,'r')
plt.plot(omega_analytical_filtered[220:300] ,'k')
# plt.plot(omega_integrated)
# plt.plot(omega_analytical)
plt.title('omega filtered')
plt.show()


#%%

omega_1D = analytical_omega(c_profile_1D)
omega_grad = omega_over_dc_dxi(c_profile_1D)

plt.scatter(c_profile_1D,omega_grad)

# plt.scatter(c_profile_1D,omega_1D)
plt.show()

#%%
plt.figure()
plt.imshow(omega_DNS[:,:,200],cmap='RdBu')
plt.title('omega_DNS')
plt.colorbar()
plt.show()

# plt.figure()
# plt.plot(omega_DNS[:,250,250])
# plt.show()


#%%
for id, c_iso in enumerate(c_iso_values):
    plt.plot(iso_dirac_reaction_rates[id,250:300,250,250])
plt.show()

iso_dirac_reaction_vec = iso_dirac_reaction_rates[:,250:300,250,250]

iso_dirac_reaction_vec_int = np.sum(iso_dirac_reaction_vec, axis=0)#trapz(iso_dirac_reaction_vec,c_iso_values,axis=0)

plt.plot(iso_dirac_reaction_vec_int)
plt.show()