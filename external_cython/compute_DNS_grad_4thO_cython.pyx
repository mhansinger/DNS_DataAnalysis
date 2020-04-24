# CYTHON Version to compute velocity gradients

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
cpdef compute_DNS_grad_4thO_cython(float[:, :, ::1] xi_np, int Nx, int Ny, int Nz, float delta_x):
    # '''
    # Compute the magnitude of the gradient of a field on the DNS grid, based on neighbour cells
    # 4th Order central differencing
    # :return: nothing
    # '''

    cdef float[:, :,::1] grad_on_DNS = np.zeros([Nx, Ny, Nz],dtype=DTYPE)  # set output array to zero

    #print('Computing gradients of xi on DNS mesh 4th Order with Cython')

    # compute gradients from the boundaries away ...
    for l in range(2,Nx-2):
        for m in range(2,Ny-2):
            for n in range(2,Nz-2):
                this_U_gradX = (-xi_np[l+2, m, n] + 8*xi_np[l+1,m, n] - 8*xi_np[l-1,m, n] + xi_np[l-2, m, n])/(12 * delta_x)
                this_U_gradY = (-xi_np[l, m+2, n] + 8*xi_np[l,m+1, n] - 8*xi_np[l,m-1, n] + xi_np[l, m-2, n])/(12 * delta_x)
                this_U_gradZ = (-xi_np[l, m, n+2] + 8*xi_np[l,m, n+1] - 8*xi_np[l,m, n-1] + xi_np[l, m, n+2])/(12 * delta_x)

                # compute the magnitude of the gradient
                this_magGrad_U = sqrt(this_U_gradX ** 2 + this_U_gradY ** 2 + this_U_gradZ ** 2)

                grad_on_DNS[l, m, n] = this_magGrad_U

    return grad_on_DNS
