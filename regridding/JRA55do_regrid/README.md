## Regridding JRA55-do 1/4-degree river discharge data to ECCO LLC90 grid

This directory contains the Python scripts used to regrid river discharge data from the JRA55-do reanalysis data (daily, 1/4-degree gridded) to weekly means on the ECCO Lat-Lon-Cap (LLC90) grid. The spatial regridding is done conservatively, so that both per-area fluxes and area-integrals of the ERA5 gridded data should be preserved in the LLC90 grid. The exception is adjacent to land where per-area fluxes are prioritized over exact area integrals, because of the different geometry of the coastline in ERA5 vs. ECCO.

There is much similarity between these codes and the ones used for ERA5 (especially the temporal regridding), but the spatial regridding has one very important difference. In ERA5 the conservation of per-area fluxes is prioritized near coastlines; in the JRA55-do codes area-integrals of the discharge are prioritized. This means that differences in coastline geometry between the JRA55-do product and ECCO LLC90 are accounted for in order to conserve the total discharge from each river. If there is no overlap between a JRA55-do river discharge cell and an ocean ECCO grid cell, the code searches for the nearest ocean ECCO grid cell to deposit the discharge into. In a few cases (e.g., Gulf of California, Sea of Okhotsk) manual adjustments have been made so that the river discharge ends up in the correct basin.

After spatial regridding, discharge from each river may be contained in one or multiple ECCO grid cells, depending on whether the source JRA55-do grid cell overlaps with multiple ECCO cells.

There are two notebooks that need to be run to accomplish the full regridding:

`jra55do_daily_to_weekly.py`: Regrids JRA55-do daily mean data to weekly means (same weekly intervals as used in the ECCO adjoint sensitivity convolutions)

`jra55do_regrid_to_llc90.py`: Regrids JRA55-do 1/4-degree river discharge data to ECCO LLC90 grid, as described above. Note that on the first run of this code, `create_indexing_weight` should be set to `True`, and a NetCDF file will be created with the weights needed to map from one grid to the other. Then on subsequent runs, set the value to `False` to greatly expedite the computation.

The other Python function in this directory `weekly_avg_jra55do_fcn.py` supports the temporal regridding script and does not need to be called directly.
