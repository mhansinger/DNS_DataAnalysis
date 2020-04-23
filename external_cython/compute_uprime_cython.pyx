# CYTHON Version to compute the SGS velocity fluctuations

#import cython
import numpy as np
cimport numpy as np
cimport cython
# from cython.parallel import prange


#################
# USE SINGLE PRECISSION (np.float32 and float in C) FOR SPEED UP
#################
DTYPE = np.float32
ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
# cpdef or cdef does not matter in speed tests
cpdef compute_U_prime_cython(float[:, :, ::1] U, float[:, :, ::1] V, float[:, :, ::1] W,
                           int Nx, int Ny, int Nz, int filter_width):
    # translate that to CYTHON
    '''

    :param U:
    :param V:
    :param W:
    :param Nx:
    :param Ny:
    :param Nz:
    :param U_bar:
    :param V_bar:
    :param W_bar:
    :param filter_width:
    :return:
    '''
    # print('\noutput_array.shape: ',output_array.shape)

    cdef float[:, :,::1] output_U = np.zeros([Nx, Ny, Nz],dtype=DTYPE)  # set output array to zero
    cdef float[:, :,::1] output_V = np.zeros([Nx, Ny, Nz],dtype=DTYPE)
    cdef float[:, :,::1] output_W = np.zeros([Nx, Ny, Nz],dtype=DTYPE)

    cdef int half_filter = filter_width / 2

    cdef float les_filter = filter_width * filter_width * filter_width

    # the indexes should be unsigned integers
    cdef unsigned int i, j, k, l, m, n

    # helper variables
    cdef float this_U_prime
    cdef float this_V_prime
    cdef float this_W_prime

    cdef float mean_U
    cdef float mean_V
    cdef float mean_W

    cdef float[:, :, ::1] this_LES_box_U
    cdef float[:, :, ::1] this_LES_box_V
    cdef float[:, :, ::1] this_LES_box_W

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
                            this_W_prime = this_W_prime + (this_LES_box_W[i, j, k] -mean_W) * \
                                           (this_LES_box_W[i, j, k] - mean_W)

                # compute c_bar of current LES box
                output_U[l, m, n] = this_U_prime/les_filter
                output_V[l, m, n] = this_V_prime/les_filter
                output_W[l, m, n] = this_W_prime/les_filter

    return output_U, output_W, output_W
