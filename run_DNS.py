# Run script for the different cases

from data_binning_PDF_simple import data_binning_PDF

print('Starting dummy case!')
filter_widths = [32]

for f in filter_widths:

    bar_dummy = data_binning_PDF(case='dummy_planar_flame', m=4.8, alpha=0.81818, beta=6, bins=20)
    bar_dummy.dask_read_transform()
    print('\nRunning with filter width: %i' % f)
    bar_dummy.run_analysis(filter_width=f, interval=4, threshold=0.02, c_rho_max=0.181, histogram=True)


