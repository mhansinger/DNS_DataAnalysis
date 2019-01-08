# Run script for the different cases

from data_binning_PDF_simple import data_binning_PDF
# to free memory
import gc
import matplotlib.pyplot as plt
import numpy as np

print('Starting dummy case!')
filter_widths = [32]

for f in filter_widths:

    bar_dummy = data_binning_PDF(case='dummy_planar_flame', m=4.5, alpha=0.81818 , beta=6, bins=50)
    bar_dummy.dask_read_transform()
    print('\nRunning with filter width: %i' % f)

    try:
        bar_dummy.run_analysis(filter_width=f, interval=5, c_min_thresh=0.00057, c_max_thresh=1, histogram=True)
    except KeyboardInterrupt:
        gc.collect()
        print('\nCANCELED BY USER!')


