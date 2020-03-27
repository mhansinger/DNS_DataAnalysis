# Check wrinkling factors like Klein

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import scipy as sp
import scipy.ndimage

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)

# read in the c- field

c_DNS = np.loadtxt('c.dat')

m = 4.4545
beta = 6
alpha = 0.81818
eps_factor=100
delta_x = 1/220
Nx = 512

# reduce the data set
print('reducing c-data set')
c_DNS = c_DNS.reshape(Nx,Nx,Nx)[int(Nx/4):-int(Nx/4),int(Nx/4):-int(Nx/4),int(Nx/4):-int(Nx/4)]

delta_th = 1.7920562525415633

#Laut Klein: Delta_LES= 4*delta_th

filter_width = int(1/(delta_x*0.7 * 50 * np.sqrt(1/1)) * 4 *delta_th)

Delta_LES = delta_x*filter_width *0.7 * 50 * np.sqrt(1/1)

#%%
def compute_gradients_4thO(c_field,Nx):
    '''4th Order gradients of c from DNS data'''

    print('Computing DNS gradients 4th Order...')

    # create empty array
    grad_c_DNS = np.zeros([Nx, Nx, Nx])

    # compute gradients from the boundaries away ...
    for l in range(2, Nx - 2):
        for m in range(2, Nx - 2):
            for n in range(2, Nx - 2):
                this_DNS_gradX = \
                    (-c_field[l + 2, m, n] + 8 * c_field[l + 1, m, n] - 8 * c_field[l - 1, m, n] + c_field[l - 2, m, n]) / (12 * delta_x)
                this_DNS_gradY = \
                    (-c_field[l, m + 2, n] + 8 * c_field[l, m + 1, n] - 8 * c_field[l, m - 1, n] + c_field[l, m - 2, n]) / (12 * delta_x)
                this_DNS_gradZ = \
                    (-c_field[l, m, n + 2] + 8 * c_field[l, m, n + 1] - 8 * c_field[l, m, n - 1] + c_field[l, m, n + 2]) / (12 * delta_x)
                # compute the magnitude of the gradient
                this_DNS_magGrad_c = np.sqrt(this_DNS_gradX ** 2 + this_DNS_gradY ** 2 + this_DNS_gradZ ** 2)

                grad_c_DNS[l, m, n] = this_DNS_magGrad_c

    return grad_c_DNS


def apply_filter(data):
    # filter c and rho data set with gauss filter function
    #print('Apply Gaussian filter...')

    # check if data is 3D array
    try:
        assert type(data) == np.ndarray
    except AssertionError:
        print('Only np.ndarrays are allowed in Gauss_filter!')

    if len(data.shape) == 1:
        data = data.reshape(Nx,Nx,Nx)

    data_filtered = sp.ndimage.filters.uniform_filter(data, [filter_width,filter_width,filter_width],mode='reflect')
    return data_filtered

#%%

c_LES = apply_filter(c_DNS)
print('c_LES filtering done...')

grad_c_DNS = compute_gradients_4thO(c_DNS,Nx=256)
print('grad_c_DNS done...')

grad_c_LES = compute_gradients_4thO(c_LES,Nx=256)
print('grad_c_LES done...')

grad_c_DNS_filtered = apply_filter(grad_c_DNS)
print('grad_c_DNS filtering done...')



#%%
# plot suff
wrinkling_factor = grad_c_DNS_filtered/(grad_c_LES+1e-8)



position = 110

plt.imshow(c_DNS[:,:,position])
plt.colorbar()
plt.title('c DNS')
plt.contour(c_DNS[:,:,position],levels=[0.1,0.7,0.9],colors='k')
plt.savefig('c_DNS_pos%i.png'%position)
plt.show()


plt.imshow(c_LES[:,:,position])
plt.colorbar()
plt.title('c LES, Delta/delta_th=4')
plt.contour(c_LES[:,:,position],levels=[0.1,0.7,0.9],colors='k')
plt.savefig('c_LES_pos%i.png'%position)
plt.show()

plt.imshow(grad_c_DNS_filtered[:,:,position],cmap='Spectral')
plt.colorbar()
plt.contour(c_LES[:,:,position],levels=[0.1,0.7,0.9])
plt.title('gradient_c DNS filtered, Delta/delta_th=4')
plt.savefig('grad_c_DNS_pos%i.png'%position)
plt.show()

plt.imshow(grad_c_LES[:,:,position],cmap='Spectral')
plt.colorbar()
plt.contour(c_LES[:,:,position],levels=[0.1,0.7,0.9])
plt.title('gradient_c LES, Delta/delta_th=4')
plt.savefig('grad_c_LES_pos%i.png'%position)
plt.show()

plt.imshow(wrinkling_factor[2:-2,2:-2,position],cmap='Spectral')
plt.colorbar()
plt.contour(c_LES[:,:,position],levels=[0.1,0.7,0.9])
plt.title('Wrinkling factor, Delta/delta_th=4')
plt.savefig('wrinkling_factor_pos%i.png'%position)
plt.show()


#%%
# histograms
plt.hist(wrinkling_factor.reshape(256**3),bins=200,range=[1.000001,5])
plt.show()
