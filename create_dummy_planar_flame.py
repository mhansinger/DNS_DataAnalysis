# this is simpy to create a planar dummy flame for testing

# @author: mhansinger
# last modified: 12.12.18

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


rho_1bar = pd.read_csv('../1bar/rho.dat',names=['rho_c'])
rho_1bar_3D = rho_1bar.values.reshape(250,250,250)
rho_by_c_1bar = pd.read_csv('../1bar/rho_by_c.dat',names=['rho_by_c'])
rho_by_c_1bar_3D = rho_by_c_1bar.values.reshape(250,250,250)
# plt.imshow(rho_by_c_1bar_3D[:,:,125])
# plt.show()
# plt.imshow(rho_1bar_3D[:,:,125])
# plt.show()

# create the dummy arrays

rho_by_c_dummy = rho_by_c_1bar_3D.copy()
rho_dummy = rho_1bar_3D.copy()


# get the min and max values
rho_min = rho_1bar.min()
rho_max = rho_1bar.max()
rho_by_c_min = rho_by_c_1bar.min()
rho_by_c_max = rho_by_c_1bar.max()


print('REWRITING THE ENTRIES!')

# AVOID LOOP!
rho_by_c_dummy[:,:,0:125].fill(float(rho_by_c_min))
rho_dummy[:, :, 0:125].fill(float(rho_min))

rho_by_c_dummy[:,:,125:].fill(float(rho_by_c_max))
rho_dummy[:, :, 125:].fill(float(rho_max))

# VERYFY WITH A PLOT
plt.imshow(rho_by_c_dummy[:,125,:])
plt.show(block=False)

# RESHAPE TO 1D VECTOR AND SHRINK DOMAIN TO 50 (INSTEAD OF 250)
rho_by_c_dummy=rho_by_c_dummy[100:150,100:150,100:150].reshape(50**3)
rho_dummy=rho_dummy[100:150,100:150,100:150].reshape(50**3)

# WRITE TO FILE
rho_by_c_dummy.tofile('rho_by_c.dat',sep='\n',format='%s')
rho_dummy.tofile('rho.dat',sep='\n',format='%s')

print('\n ALL DONE!')
