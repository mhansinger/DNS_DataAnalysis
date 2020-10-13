# CYTHON Version to compute velocity gradients

# use: cythonize -a -i compute_gradU_LES_4thO_cython.pyx

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
cpdef compute_gradU_LES_4thO_tensor_cython(float[:, :, ::1] U_bar, float[:, :, ::1] V_bar, float[:, :, ::1] W_bar, int Nx, int Ny, int Nz, float delta_x, int filter_width):
    # '''
    # Compute the magnitude of the gradient of the LES U-field, based on neighbour cells
    # 4th Order central differencing
    # :return: nothing
    # '''

    cdef float[:, :,::1] grad_U_x = np.zeros([Nx, Ny, Nz],dtype=DTYPE)  # set output array to zero
    cdef float[:, :,::1] grad_V_x = np.zeros([Nx, Ny, Nz],dtype=DTYPE)
    cdef float[:, :,::1] grad_W_x = np.zeros([Nx, Ny, Nz],dtype=DTYPE)

    cdef float[:, :,::1] grad_U_y = np.zeros([Nx, Ny, Nz],dtype=DTYPE)  # set output array to zero
    cdef float[:, :,::1] grad_V_y = np.zeros([Nx, Ny, Nz],dtype=DTYPE)
    cdef float[:, :,::1] grad_W_y = np.zeros([Nx, Ny, Nz],dtype=DTYPE)

    cdef float[:, :,::1] grad_U_z = np.zeros([Nx, Ny, Nz],dtype=DTYPE)  # set output array to zero
    cdef float[:, :,::1] grad_V_z = np.zeros([Nx, Ny, Nz],dtype=DTYPE)
    cdef float[:, :,::1] grad_W_z = np.zeros([Nx, Ny, Nz],dtype=DTYPE)

    # the indexes should be unsigned integers
    cdef unsigned int l, m, n

    print('Computing gradients of U_bar on DNS mesh 4th Order with Cython')

    # compute gradients from the boundaries away ...
    for l in range(2*filter_width,Nx-2*filter_width):
        for m in range(2*filter_width,Ny-2*filter_width):
            for n in range(2*filter_width,Nz-2*filter_width):
                grad_U_x[l,m,n] = (-U_bar[l+2*filter_width, m, n] + 8*U_bar[l+1*filter_width,m, n] - 8*U_bar[l-1*filter_width,m, n] + U_bar[l-2*filter_width, m, n])/(12 * delta_x*filter_width)
                grad_V_y[l,m,n] = (-U_bar[l, m+2*filter_width, n] + 8*U_bar[l,m+1*filter_width, n] - 8*U_bar[l,m-1*filter_width, n] + U_bar[l, m-2*filter_width, n])/(12 * delta_x*filter_width)
                grad_W_z[l,m,n] = (-U_bar[l, m, n+2*filter_width] + 8*U_bar[l,m, n+1*filter_width] - 8*U_bar[l,m, n-1*filter_width] + U_bar[l, m, n-2*filter_width])/(12 * delta_x*filter_width)

                grad_U_x[l,m,n] = (-V_bar[l+2*filter_width, m, n] + 8*V_bar[l+1*filter_width,m, n] - 8*V_bar[l-1*filter_width,m, n] + V_bar[l-2*filter_width, m, n])/(12 * delta_x*filter_width)
                grad_V_y[l,m,n] = (-V_bar[l, m+2*filter_width, n] + 8*V_bar[l,m+1*filter_width, n] - 8*V_bar[l,m-1*filter_width, n] + V_bar[l, m-2*filter_width, n])/(12 * delta_x*filter_width)
                grad_W_z[l,m,n] = (-V_bar[l, m, n+2*filter_width] + 8*V_bar[l,m, n+1*filter_width] - 8*V_bar[l,m, n-1*filter_width] + V_bar[l, m, n-2*filter_width])/(12 * delta_x*filter_width)

                grad_U_x[l,m,n] = (-W_bar[l+2*filter_width, m, n] + 8*W_bar[l+1*filter_width,m, n] - 8*W_bar[l-1*filter_width,m, n] + W_bar[l-2*filter_width, m, n])/(12 * delta_x*filter_width)
                grad_V_y[l,m,n] = (-W_bar[l, m+2*filter_width, n] + 8*W_bar[l,m+1*filter_width, n] - 8*W_bar[l,m-1*filter_width, n] + W_bar[l, m-2*filter_width, n])/(12 * delta_x*filter_width)
                grad_W_z[l,m,n] = (-W_bar[l, m, n+2*filter_width] + 8*W_bar[l,m, n+1*filter_width] - 8*W_bar[l,m, n-1*filter_width] + W_bar[l, m, n-2*filter_width])/(12 * delta_x*filter_width)

    return grad_U_x, grad_V_x, grad_W_x, grad_U_y, grad_V_y, grad_W_y, grad_U_z, grad_V_z, grad_W_z
