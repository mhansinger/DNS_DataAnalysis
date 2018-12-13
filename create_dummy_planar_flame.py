# this is simply to create a planar dummy flame for code validation
# is created from 1bar case

# @author: mhansinger
# last modified: 12.12.18

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

N_DNS = 250

rho_1bar = pd.read_csv('../1bar/rho.dat',names=['rho_c'])
rho_1bar_3D = rho_1bar.values.reshape(N_DNS,N_DNS,N_DNS)
rho_by_c_1bar = pd.read_csv('../1bar/rho_by_c.dat',names=['rho_by_c'])
rho_by_c_1bar_3D = rho_by_c_1bar.values.reshape(N_DNS,N_DNS,N_DNS)

# create the dummy arrays

rho_by_c_dummy = rho_by_c_1bar_3D.copy()
rho_dummy = rho_1bar_3D.copy()

# get the min and max values
rho_min = float(rho_1bar.min())
rho_max = float(rho_1bar.max())
rho_by_c_min = float(rho_by_c_1bar.min())
rho_by_c_max = float(rho_by_c_1bar.max())

#READ IN ARTIFICIAL C_PROFIL AND SCALE IT TO RHO_MIN RHO_MAX...
c_profile = np.loadtxt('C_verlauf.txt')
profile_len=len(c_profile)

rho_profile = (1-c_profile)*(rho_max-rho_min) +rho_min
rho_by_c_profile = c_profile*(rho_by_c_max-rho_by_c_min) + rho_by_c_min

print('REWRITING THE ENTRIES!')

# AVOID LOOP!
rho_by_c_dummy[:,:,0:int(N_DNS/2)].fill(float(rho_by_c_min))
rho_dummy[:, :, int(N_DNS/2):].fill(float(rho_min))

rho_by_c_dummy[:,:,int(N_DNS/2):].fill(float(rho_by_c_max))
rho_dummy[:, :, :int(N_DNS/2)].fill(float(rho_max))

# c_dummy = rho_dummy.copy()
# c_dummy[:,:,:125].fill(0)
# c_dummy[:,:,125:].fill(1)

#FILL WITH THE C_PROFILE SCALED ENTRIES
print('Fill with the c-profile entries')

start_id = 125
for i in range(profile_len):
    rho_by_c_dummy[:, :, i+start_id] = rho_by_c_profile[i]
    rho_dummy[:, :, i + start_id] = rho_profile[i]
    #c_dummy[:, :, i + start_id] = c_profile[i]

new_domain = 250

# RESHAPE TO 1D VECTOR AND SHRINK DOMAIN TO 50 (INSTEAD OF 250)
rho_by_c_dummy=rho_by_c_dummy.reshape(new_domain**3)
rho_dummy=rho_dummy.reshape(new_domain**3)

# WRITE TO FILE
rho_by_c_dummy.tofile('rho_by_c.dat',sep='\n',format='%s')
rho_dummy.tofile('rho.dat',sep='\n',format='%s')

# VERYFY WITH A PLOT
#plt.close('all')

#plt.figure()
#plt.imshow(rho_dummy.reshape(new_domain,new_domain,new_domain)[:,10,:])
#plt.title('rho dummy')
#plt.colorbar()

#plt.figure()
#plt.imshow(rho_by_c_dummy.reshape(new_domain,new_domain,new_domain)[:,10,:])
#plt.title('rho by c dummy')
#plt.colorbar()

c_dummy=rho_by_c_dummy.reshape(new_domain,new_domain,new_domain) / rho_dummy.reshape(new_domain,new_domain,new_domain)

#plt.figure()
#plt.imshow(c_dummy[:,10,:])#c_dummy[50:200,125,50:200])
#plt.title('c dummy')
#plt.colorbar()

# plt.figure()
# plt.imshow(rho_1bar_3D[50:200,125,50:200])
# plt.title('rho 1bar')
# plt.colorbar()

# plt.figure()
# plt.imshow(rho_by_c_1bar_3D[50:200,125,50:200])
# plt.title('rho by c 1bar')
# plt.colorbar()
#
# plt.figure()
# plt.imshow(rho_by_c_1bar_3D[50:200,125,50:200]/rho_1bar_3D[50:200,125,50:200])
# plt.title('c 1bar')
# plt.colorbar()

#plt.figure()
#plt.title('rho_profile')
#plt.plot(rho_profile)

#plt.figure()
#plt.title('rho_by_c_dummy_profile')
#plt.plot(rho_by_c_profile)

#plt.figure()
#plt.title('c_dummy_profile')
#plt.plot(c_dummy[10,12,:] )

# plt.figure()
# plt.title('rho_profile')
# plt.plot(rho_1bar_3D[50:200,100,50])
#
# plt.figure()
# plt.title('rho_by_c_profile')
# plt.plot(rho_by_c_1bar_3D[50:200,100,50])

#plt.show(block=False)

print('\nALL DONE!')
