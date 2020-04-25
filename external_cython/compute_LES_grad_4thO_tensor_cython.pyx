# CYTHON Version to compute velocity gradients

#use: cythonize -a -i compute_DNS_grad_4thO_cython.pyx

#import cython
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
# from cython.parallel import prange

#################
# USE SINGLE PRECISSION (np.float32 and float in C) FOR SPEED UP
#################
DTYPE = np.float32
ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
# cpdef or cdef does not matter in speed tests
cpdef compute_LES_grad_4thO_tensor_cython(float[:, :, ::1] field, int Nx, int Ny, int Nz, float delta_x, int filter_width):
    # '''
    # Compute the magnitude of the gradient of a field on the DNS grid, based on neighbour cells
    # 4th Order central differencing
    # :return: nothing
    # '''

    # the indexes should be unsigned integers
    cdef unsigned int l, m, n

    cdef float[:, :,::1] grad_X = np.zeros([Nx, Ny, Nz],dtype=DTYPE)  # set output array to zero
    cdef float[:, :,::1] grad_Y = np.zeros([Nx, Ny, Nz],dtype=DTYPE)  # set output array to zero
    cdef float[:, :,::1] grad_Z = np.zeros([Nx, Ny, Nz],dtype=DTYPE)  # set output array to zero

    #print('Computing gradients of xi on DNS mesh 4th Order with Cython')

    # compute gradients from the boundaries away ...
    for l in range(2*filter_width,Nx-2*filter_width):
        for m in range(2*filter_width,Ny-2*filter_width):
            for n in range(2*filter_width,Nz-2*filter_width):
                grad_X[l,m,n] = (-field[l+2*filter_width, m, n] + 8*field[l+1*filter_width,m, n] - 8*field[l-1*filter_width,m, n] + field[l-2*filter_width, m, n])/(12 * filter_width* delta_x)
                grad_Y[l,m,n] = (-field[l, m+2*filter_width, n] + 8*field[l,m+1*filter_width, n] - 8*field[l,m-1*filter_width, n] + field[l, m-2*filter_width, n])/(12 * filter_width* delta_x)
                grad_Z[l,m,n] = (-field[l, m, n+2*filter_width] + 8*field[l,m, n+1*filter_width] - 8*field[l,m, n-1*filter_width] + field[l, m, n-2*filter_width])/(12 * filter_width* delta_x)

    return grad_X, grad_Y, grad_Z
