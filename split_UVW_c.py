# compute c.dat from rho_c.dat and rho.dat
# compute U.dat V.dat W.dat

import numpy as np

#read

rho_by_c = np.loadtxt('rho_by_c.dat')
rho = np.loadtxt('rho.dat')

c = rho_by_c/rho

np.savetxt('c.dat',c)

del c, rho, rho_by_c

UVW = np.loadtxt('UVW.dat')

np.savetxt('U.dat',UVW[:,0])
np.savetxt('V.dat',UVW[:,1])
np.savetxt('W.dat',UVW[:,2])

