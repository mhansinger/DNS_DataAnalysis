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

c_low = 0.01
c_high = 0.99
colormap = 'rainbow' #'seismic'#'bwr'#'jet'
Re = 50

m = 4.454

delta_dx = 1/220

d_th = (m + 1) ** (1 / m + 1) / m

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

    print(file_name)

    # reduce to c_bar > 0.5
    data_org = data
    data = data[data.c_bar > c_low]

    # PLOTS
    plt.figure(figsize=(5, 5))
    c_max = max(data.omega_DNS_filtered) * 1.2
    plt.plot([0, c_max], [0, c_max], '--k', lw=0.5)
    plt.scatter(data.omega_DNS_filtered, data.omega_model, s=2, c=data.c_bar, cmap=plt.cm.get_cmap(colormap),rasterized=False)
    # clb = plt.colorbar()
    plt.clim(c_low, c_high)
    # clb.ax.set_title(r'$\overline{c}$')
    plt.xlim(0, c_max)
    plt.ylim(0, c_max)
    # plt.xlabel(r'$\overline{\omega}_{DNS}$')
    # plt.ylabel(r'$\overline{\omega}_m$')
    Delta_LES = round(filter_widths[id] * delta_dx * Re*0.7, 3)
    plt.title(r'$\Delta_{LES}=%s\delta_{th}$ / $n=%s$' % (str(round(Delta_LES/d_th,2)),str(filter_widths[id])))
    R2 = round(pearsonr(data.omega_DNS_filtered, data.omega_model)[0],3)
    plt.text(c_max*0.1, c_max*0.85 , r'$R^2=%s$' % str(R2))
    plot_name = join(dir,'plots','scatter_Delta_%s.png' % (str(filter_widths[id])))
    plt.savefig(plot_name)
    plt.show()


    plt.figure(figsize=(5, 5))
    c_max = max(data.omega_DNS_filtered) * 1.2
    if c_max > 5:
        c_max = 5
    plt.plot([0, c_max], [0, c_max], '--k', lw=0.5)
    plt.scatter(data.omega_DNS_filtered, data.omega_model * data.wrinkling, s=2, c=data.c_bar,
                cmap=plt.cm.get_cmap(colormap),rasterized=False)
    # clb = plt.colorbar()
    # clb.ax.set_title(r'$\overline{c}$')
    plt.clim(c_low, c_high)
    # plt.xlabel(r'$\overline{\omega}_{DNS}$')
    # plt.ylabel(r'$\overline{\omega}_m \cdot \Xi_{grad}$')
    plt.xlim(0, c_max)
    plt.ylim(0, c_max)
    Delta_LES = round(filter_widths[id] * delta_dx * Re*0.7, 3)
    plt.title(r'$\Delta_{LES}=%s\delta_{th}$ / $n=%s$' % (str(round(Delta_LES/d_th,2)),str(filter_widths[id])))
    R2 = round(pearsonr(data.omega_DNS_filtered, data.omega_model*data.wrinkling)[0],3)
    if R2 == np.nan:
        R2 = 0
    plt.text(c_max*0.1, c_max*0.85 , r'$R^2=%s$' % str(R2))
    plot_name = join(dir,'plots','scatter_Delta_%s_wrinkling.png' % (str(filter_widths[id])))
    plt.savefig(plot_name)
    plt.show()


    # omega * iso wrinkling
    plt.figure(figsize=(5, 5))
    c_max = max(data.omega_model * data.isoArea) * 1.2
    plt.plot([0, c_max], [0, c_max], '--k', lw=0.5)
    plt.scatter(data.omega_DNS_filtered, data.omega_model * data.isoArea, s=2, c=data.c_bar,
                cmap=plt.cm.get_cmap(colormap),rasterized=False)
    # clb = plt.colorbar()
    plt.clim(c_low, c_high)
    # clb.ax.set_title(r'$\overline{c}$')
    # plt.xlabel(r'$\overline{\omega}_{DNS}$')
    # plt.ylabel(r'$\overline{\omega}_m \cdot \Xi_{iso}$')
    plt.xlim(0, c_max)
    plt.ylim(0, c_max)
    Delta_LES = round(filter_widths[id] *delta_dx * Re* 0.7, 3)
    plt.title(r'$\Delta_{LES}=%s\delta_{th}$ / $n=%s$' % (str(round(Delta_LES/d_th,2)),str(filter_widths[id])))
    R2 = round(pearsonr(data.omega_DNS_filtered, data.omega_model* data.isoArea)[0],3)
    plt.text(c_max*0.1, c_max*0.85 , r'$R^2=%s$' % str(R2))
    plot_name = join(dir,'plots','scatter_Delta_%s_isoArea.png' % (str(filter_widths[id])))
    plt.savefig(plot_name)
    plt.show()


    # wrinkling factor vs. isoArea
    plt.figure(figsize=(5, 5))
    plt.plot([1, 3], [1, 3], '--k')
    plt.scatter(data.wrinkling, data.isoArea, s=2, c=data.c_bar, cmap=plt.cm.get_cmap(colormap),rasterized=False)
    # clb = plt.colorbar()
    # clb.ax.set_title(r'$\overline{c}$')
    # plt.xlabel(r'$\Xi_{grad}$')
    # plt.ylabel(r'$\Xi_{iso}$')
    plt.xlim(1, min([max(data.wrinkling)*1.05,10]))
    plt.ylim(1, max(data.isoArea)*1.05)
    #plt.title(r'$\Xi_{grad}$ vs. $\Xi_{iso}$: $\Delta_{LES}=%s$ / $n=%s$' % (str(Delta_LES),str(filter_widths[id])))
    plt.title(r'$\Delta_{LES}=%s\delta_{th}$ / $n=%s$' % (str(round(Delta_LES/d_th,2)),str(filter_widths[id])))
    R2 = round(pearsonr(data.wrinkling, data.isoArea)[0],3)
    plot_name = join(dir,'plots','scatter_Delta_%s_wrinkl_vs_isoArea.png' % (str(filter_widths[id])))
    plt.savefig(plot_name)
    plt.show()


    # isoArea wrinkling over c_bar
    plt.figure(figsize=(5, 5))
    plt.scatter(data_org.c_bar, data_org.isoArea, s=2, c='k')
    # plt.xlabel(r'$\overline{c}$')
    # plt.ylabel(r'$\Xi_{iso}$')
    plt.xlim(0, 1)
    plt.ylim(0, max(data.isoArea)*1.05)
    plt.title(r'$\Delta_{LES}=%s\delta_{th}$ / $n=%s$' % (str(round(Delta_LES/d_th,2)),str(filter_widths[id])))
    R2 = round(pearsonr(data.wrinkling, data.isoArea)[0],3)
    plot_name = join(dir,'plots','scatter_Delta_%s_cbar_isoArea.png' % (str(filter_widths[id])))
    plt.savefig(plot_name)
    plt.show()


    # # isoArea vs c_bar
    # plt.figure(figsize=(6, 5))
    # plt.scatter(data.c_bar, data.isoArea, s=2, c='k')
    # plt.xlabel(r'$\overline{c}$')
    # plt.ylabel(r'$\Xi_{iso}$')
    # plt.ylim(1, 3)
    # plt.xlim(0, 1)
    # # plt.plot([1,3],[1,3],'--k')
    # plt.title(r'$\Xi_{grad}$ vs. $\overline{c}$: $\Delta_{LES}=%s$ / $n=%s$' % (str(Delta_LES),str(filter_widths[id])))
    # # plt.savefig('scatter_isoArea_vs_cBar_%s_%s_analytical.png' % (filter_type, str(filter_width)))
    # plt.show()

    #del data
