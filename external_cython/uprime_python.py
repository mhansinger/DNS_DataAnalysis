import numpy as np
import scipy.ndimage
from numba import jit, cuda

def compute_U_prime_alternative( U, V, W, Nx, Ny, Nz, filter_width):
    # translate that to CYTHON

    # print('\noutput_array.shape: ',output_array.shape)

    output_U = np.zeros((Nx, Ny, Nz))  # set output array to zero
    output_V = np.zeros((Nx, Ny, Nz))
    output_W = np.zeros((Nx, Ny, Nz))

    half_filter = int(filter_width / 2)

    les_filter = filter_width * filter_width * filter_width

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
                for i in range(0, filter_width):
                    for j in range(0, filter_width):
                        for k in range(0, filter_width):
                            mean_U = mean_U + this_LES_box_U[i, j, k]
                            mean_V = mean_V + this_LES_box_V[i, j, k]
                            mean_W = mean_W + this_LES_box_W[i, j, k]

                mean_U = mean_U / les_filter
                mean_V = mean_V / les_filter
                mean_W = mean_W / les_filter
                # compute the variance of each cell

                for i in range(0, filter_width):
                    for j in range(0, filter_width):
                        for k in range(0, filter_width):
                            this_U_prime = this_U_prime + (this_LES_box_U[i, j, k] - mean_U) * \
                                           (this_LES_box_U[i, j, k] - mean_U)
                            this_V_prime = this_V_prime + (this_LES_box_V[i, j, k] - mean_V) * \
                                           (this_LES_box_V[i, j, k] - mean_V)
                            this_W_prime = this_W_prime + (this_LES_box_W[i, j, k] - mean_W) * \
                                           (this_LES_box_W[i, j, k] - mean_W)

                # compute c_bar of current LES box
                output_U[l, m, n] = this_U_prime/ les_filter
                output_V[l, m, n] = this_V_prime/ les_filter
                output_W[l, m, n] = this_W_prime/ les_filter

    return output_U, output_W, output_W




def compute_U_prime_U_bar(U, V, W, Nx, Ny, Nz, U_bar, V_bar, W_bar, filter_width):
    # translate that to CYTHON

    # print('\noutput_array.shape: ',output_array.shape)

    output_U = np.zeros((Nx, Ny, Nz))  # set output array to zero
    output_V = np.zeros((Nx, Ny, Nz))
    output_W = np.zeros((Nx, Ny, Nz))

    half_filter = int(filter_width / 2)

    les_filter = filter_width * filter_width * filter_width

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

                this_U_prime = 0.0
                this_V_prime = 0.0
                this_W_prime = 0.0

                for i in range(0, filter_width):
                    for j in range(0, filter_width):
                        for k in range(0, filter_width):
                            this_U_prime = this_U_prime + (this_LES_box_U[i, j, k] - U_bar[i,j,k]) * \
                                           (this_LES_box_U[i, j, k] - U_bar[i,j,k])
                            this_V_prime = this_V_prime + (this_LES_box_V[i, j, k] - V_bar[i,j,k]) * \
                                           (this_LES_box_V[i, j, k] - V_bar[i,j,k])
                            this_W_prime = this_W_prime + (this_LES_box_W[i, j, k] - W_bar[i,j,k]) * \
                                           (this_LES_box_W[i, j, k] - W_bar[i,j,k])

                # compute c_bar of current LES box
                output_U[l, m, n] = this_U_prime
                output_V[l, m, n] = this_V_prime
                output_W[l, m, n] = this_W_prime

    return output_U, output_W, output_W

######
# original implementation
def compute_U_prime(U, V, W, filter_width):
    '''
    # compute the SGS velocity fluctuations
    # u_prime is the STD of the U-DNS components within a LES cell
    :return: nothing
    '''
    #print('Computing U prime')
    U_prime = scipy.ndimage.generic_filter(U, np.var, mode='wrap',
                                                size=(filter_width, filter_width,filter_width))
    #print('Computing V prime')
    V_prime = scipy.ndimage.generic_filter(V, np.var, mode='wrap',
                                                size=(filter_width, filter_width, filter_width))
    #print('Computing W prime')
    W_prime = scipy.ndimage.generic_filter(W, np.var, mode='wrap',
                                                size=(filter_width, filter_width, filter_width))

    return U_prime, V_prime, W_prime