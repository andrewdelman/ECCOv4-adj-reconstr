## Regridding ERA5 1/4-degree surface flux data to ECCO LLC90 grid

This directory contains the Python scripts used to regrid ERA5 surface flux reanalysis data (hourly, 1/4-degree gridded) to weekly means on the ECCO Lat-Lon-Cap (LLC90) grid. The spatial regridding is done conservatively, so that both per-area fluxes and area-integrals of the ERA5 gridded data should be preserved in the LLC90 grid. The exception is adjacent to land where per-area fluxes are prioritized over exact area integrals, because of the different geometry of the coastline in ERA5 vs. ECCO.

The spatial regridding is also aware of the land/ocean masks in both ERA5 and ECCO, so that flux values over ERA5 land grid cells are excluded from ECCO ocean grid cells. This is important because ERA5 surface fluxes may be very different between adjacent land and ocean grid cells. There is an exception: the land/ocean masks are ignored when there are no ERA5 ocean cells overlapping with ECCO ocean cells--this usually only happens in the polar regions where ice coverage may be quite different between the two models.

There are three notebooks that need to be run to accomplish the full regridding:

`era5_hourly_to_daily.py`: Regrids ERA5 hourly data to daily means

`era5_daily_to_weekly.py`: Regrids ERA5 daily mean data to weekly means (same weekly intervals as used in the ECCO adjoint sensitivity convolutions)

`era5_regrid_to_llc90.py`: Regrids ERA5 1/4-degree data to ECCO LLC90 grid, as described above. Note that on the first run of this code, `create_indexing_weight` should be set to `True`, and a NetCDF file will be created with the weights needed to map from one grid to the other. Then on subsequent runs, set the value to `False` to greatly expedite the computation.

The other two Python functions in this directory ending in `fcn.py` are supporting modules called by the temporal regridding scripts above.
