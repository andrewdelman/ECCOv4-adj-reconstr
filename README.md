## Summary

This repository contains Jupyter notebooks and supporting Python modules to compute the convolution of ECCOv4r4 adjoint sensitivities and surface fluxes (from ECCOv4r4, ERA5, JRA55-do river discharge). These codes were used to generate the figures in the following paper:

Delman, A., Wang, O., Lee, T., Forcing of subannual-to-decadal sea level variability and the recent rise along the U.S. Gulf Coast. J. Geophys. Res. Oceans.

The codes are currently set up to reconstruct sea level in regional averages along the eastern and western Gulf Coasts, or at select tide gauge sites along the Gulf Coast. But with different sets of adjoint sensitivities (can be generated using the ECCO Modeling Utilities) and by changing the reference/obs time series used for validation, these codes could be modified to reconstruct local values or regional averages of any of these ocean state variables: T, S, u, v, SSH, OBP.

The convolution computation is done by the notebook `forcsens_contribs_stats_compute.ipynb`. This notebook archives the results of the computation (contributions from each forcing type, lead time, and individual model grid cell or aggregated "cells"/regions) as a `zarr` store. Running the convolution computation is a prerequisite for most of the notebooks in the `analysis` directory.


## Notebooks used to generate figures

All of the following notebooks are in the `analysis` directory. Unless otherwise indicated, they require as input a `zarr` store with can be computed using the convolution notebook `forcsens_contribs_stats_compute.ipynb`.

Delman, A., Wang, O., Lee, T., Forcing of subannual-to-decadal sea level variability and the recent rise along the U.S. Gulf Coast. J. Geophys. Res. Oceans.

Figure 1: `adjoint_sens_read.py`, does not require convolution notebook to be run before

Figure 2: `altimetry_stateest_compare.ipynb`, does not require convolution notebook to be run before

Figures 3 and 4: `tseries_trend_comparisons.ipynb`

Figures 5 and 6: `pred_varexp_forcsens_var_cells_byfreq_plot.ipynb`

Figures 7 and 8: `pred_varexp_region_plot.ipynb` for Fig. 7a, `Region_tseries_plot.ipynb` for the other panels

Figure 9: `tseries_trend_comparisons.ipynb`

Figure 10: `Region_tseries_plot.ipynb`

Figure 11: `pred_varexp_forcsens_var_cells_byfreq_plot.ipynb`

Figure 12: `Region_tseries_plot.ipynb`
