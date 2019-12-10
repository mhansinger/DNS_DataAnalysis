'''
Code to generate plots based on the DNS scatter data
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

case = 'UPRIME5' #'UPRIME15'

path_to_data = '/media/max/HDD2/HDD2_Data/DNS_Data_Klein/Planar/NX512'

dir = join(path_to_data,case)

files_in_dir = os.listdir(dir)

scatter_files = [f for f in files_in_dir if f.endswith('.csv')]

filter_widths = [int(f.split('_')[-2]) for f in scatter_files]


for id, file_name in enumerate(scatter_files):

    this_file_path =join(dir,file_name)

    data = pd.read_csv(this_file_path)
    data = data.sample(frac=1.0)
    # data.columns
    print('c_max: ', data.c_bar.max())
    print('c_min: ', data.c_bar.min())

    plt.figure(figsize=(6, 5))
    plt.scatter(data.omega_DNS_filtered, data.omega_model, s=2, c=data.c_bar, cmap=plt.cm.get_cmap('RdBu'))
    clb = plt.colorbar()
    clb.ax.set_title(r'$\overline{c}$')
    c_max = max(data.omega_DNS_filtered) * 1.2
    plt.xlim(0, c_max)
    plt.ylim(0, c_max)
    plt.plot([0, c_max], [0, c_max], '--k')
    plt.xlabel(r'$\overline{\omega}_{DNS}$')
    plt.ylabel(r'$\overline{\omega}_m$')
    plt.title(r'Filter width: %s DNS points' % (str(filter_widths[id])))
    #plt.savefig('scatter_filter_%s_%s_analytical.png' % (filter_type, str(filter_width)))
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(data.omega_DNS_filtered, data.omega_model * data.wrinkling, s=2, c=data.c_bar,
                cmap=plt.cm.get_cmap('RdBu'))
    clb = plt.colorbar()
    clb.ax.set_title(r'$\overline{c}$')
    plt.xlabel(r'$\overline{\omega}_{DNS}$')
    plt.ylabel(r'$\overline{\omega}_m \cdot \Xi$')
    c_max = max(data.omega_model * data.wrinkling) * 1.2
    if c_max > 5:
        c_max = 5
    plt.xlim(0, c_max)
    plt.ylim(0, c_max)
    plt.plot([0, c_max], [0, c_max], '--k', lw=0.5)
    plt.title(r'Filter width: %s DNS points' % (str(filter_widths[id])))
    #plt.savefig('scatter_filter_scaled_%s_%s_analytical.png' % (filter_type, str(filter_width)))
    plt.show()

    # omega * iso wrinkling
    plt.figure(figsize=(6, 5))
    plt.scatter(data.omega_DNS_filtered, data.omega_model * data.isoArea, s=2, c=data.c_bar,
                cmap=plt.cm.get_cmap('RdBu'))
    clb = plt.colorbar()
    clb.ax.set_title(r'$\overline{c}$')
    plt.xlabel(r'$\overline{\omega}_{DNS}$')
    plt.ylabel(r'$\overline{\omega}_m \cdot \Xi_{iso}$')
    c_max = max(data.omega_model * data.isoArea) * 1.2
    plt.xlim(0, c_max)
    plt.ylim(0, c_max)
    plt.plot([0, c_max], [0, c_max], '--k', lw=0.5)
    plt.title(r'Filter width: %s DNS points' % (str(filter_widths[id])))
    #plt.savefig('scatter_filter_scaled_%s_%s_analytical.png' % (filter_type, str(filter_width)))
    plt.show()

    #del data
