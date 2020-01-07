'''
plot from the DNS data the different omegas
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import os
from scipy.stats import pearsonr
# matplotlib.rcParams['text.usetex'] = True

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

colormap = 'rainbow' #'seismic'#'bwr'#'jet'

# path to data
dir_path = '/media/max/HDD2/HDD2_Data/DNS_Data_Klein/Planar/NX512/UPRIME15/filtered_data'

filter_width = 16

file_name = 'omega_filtered_modeled_%i_nth1.csv' % filter_width

data_path = join(dir_path,file_name)

# read in the file
DNS_pd = pd.read_csv(data_path)

#remove nan and inf
DNS_pd= DNS_pd.fillna(0)
DNS_pd= DNS_pd.replace(np.inf,0)

#%%
print(DNS_pd.columns)

omega_DNS = DNS_pd['omega_DNS'].values.reshape(512,512,512)
omega_filtered = DNS_pd['omega_filtered'].values.reshape(512,512,512)
omega_by_iso = DNS_pd['omega_model_by_isoArea'].values.reshape(512,512,512)
omega_by_grad = DNS_pd['omega_model_by_wrinkling'].values.reshape(512,512,512)
c_bar = DNS_pd['c_bar'].values.reshape(512,512,512)

#%%
colormap = 'seismic' # bwr'#'jet' #'rainbow' #'seismic'#'bwr'#'jet'

x_pos = 230

# plots
fig=plt.figure(figsize=(5, 5),frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(omega_DNS[x_pos,40:472,40:472],cmap=plt.cm.get_cmap(colormap))
plot_name = join(dir_path,'omega_DNS_x_pos%s_n%i.png' % (str(x_pos),filter_width))
fig.savefig(plot_name)
plt.show()


fig=plt.figure(figsize=(5, 5),frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(omega_filtered[x_pos,40:472,40:472],cmap=plt.cm.get_cmap(colormap))
# plt.clim(0, omega_DNS.max())
# plt.axis('off')
plot_name = join(dir_path,'omega_filtered_x_pos%s_n%i.png' % (str(x_pos),filter_width))
fig.savefig(plot_name)
plt.show()


fig=plt.figure(figsize=(5, 5),frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(omega_by_iso[x_pos,40:472,40:472],cmap=plt.cm.get_cmap(colormap))
# plt.clim(0, omega_DNS.max())
# # plt.title(r'$\overline{\omega}_m \cdot \Xi_{iso}$')
# plt.axis('off')
plot_name = join(dir_path,'omega_iso_x_pos%s_n%i.png' % (str(x_pos),filter_width))
fig.savefig(plot_name)
plt.show()


fig=plt.figure(figsize=(5, 5),frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(c_bar[x_pos,40:472,40:472],cmap=plt.cm.get_cmap('viridis'))
# clb=plt.colorbar()
# clb.ax.set_title(r'$\omega$')
# plt.clim(0, c_bar.max())
# plt.title(r'$\overline{\omega}_m \cdot \Xi_{iso}$')
# plt.axis('off')
plot_name = join(dir_path,'c_bar_x_pos%s_n%i.png' % (str(x_pos),filter_width))
fig.savefig(plot_name)
plt.show()


# plt.imshow(omega_by_grad[x_pos,40:472,40:472],cmap=plt.cm.get_cmap(colormap))
# plt.colorbar()
# plt.clim(0, omega_DNS.max())
# plt.title(r'$\overline{\omega}_m \cdot \Xi_{grad}$')
# plot_name = join(dir_path,'omega_grad_x_pos%s_n%i.png' % (str(x_pos),filter_width))
# plt.savefig(plot_name)
# plt.show()
#
# # plot error
# plt.imshow(abs(omega_filtered[x_pos,40:472,40:472]-omega_by_iso[x_pos,40:472,40:472]),cmap=plt.cm.get_cmap(colormap))
# plt.colorbar()
# plt.clim(0, omega_DNS.max())
# plt.title(r'error: $\overline{\omega}_{DNS}$ - $\overline{\omega}_m \cdot \Xi_{iso}$')
# plot_name = join(dir_path,'omega_error_iso_x_pos%s_n%i.png' % (str(x_pos),filter_width))
# plt.savefig(plot_name)
# plt.show()
#
# # plot error
# plt.imshow(abs(omega_filtered[x_pos,40:472,40:472]-omega_by_grad[x_pos,40:472,40:472]),cmap=plt.cm.get_cmap(colormap))
# plt.colorbar()
# plt.clim(0, omega_DNS.max())
# plt.title(r'error: $\overline{\omega}_{DNS}$ - $\overline{\omega}_m \cdot \Xi_{grad}$')
# plot_name = join(dir_path,'omega_error_grad_x_pos%s_n%i.png' % (str(x_pos),filter_width))
# plt.savefig(plot_name)
# plt.show()