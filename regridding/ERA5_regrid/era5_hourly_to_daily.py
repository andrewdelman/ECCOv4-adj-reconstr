# create daily mean ERA5 files
# from hourly mean ERA5 files (spanning a month)

import numpy as np
import xarray as xr
from os.path import join,expanduser
import glob

import dask.array as da

import multiprocessing
from daily_avg_era5_fcn import *

## Year range to process (unlike usual Python convention, year_last is included in the range)
year_first = 2013
year_last = 2018


## Path and fileform (start of filenames) for hourly ERA5 data
source_filepath = join('/nobackup','adelman','ERA5','hourly')
source_fileform = 'era5_sflux_hourly_'
source_filepathform = join(source_filepath,source_fileform)

## Path and fileform for output daily ERA5 data
output_filepath = join('/nobackup','adelman','ERA5','daily')
output_fileform = 'era5_sflux_daily_'
output_filepathform = join(output_filepath,output_fileform)


for year in range(year_first,year_last+1):
    year_str = str(year)
    
    # files_in_year = sorted(glob.glob(source_filepathform+year_str+'_??.nc'))
    files_in_year = sorted(glob.glob(source_filepathform+year_str+'-??.nc'))
    for filenum_inyear,source_file in enumerate(files_in_year):
        month_str = source_file[-5:-3]
        ds_src = xr.open_mfdataset(source_file)
        data_var_names = list(ds_src.keys())
        
        if 'expver' in ds_src.coords:
            if 'expver' in ds_src.dims:
                ds_new = ds_src.drop_dims('expver')
            else:
                ds_new = ds_src.drop_vars('expver')
        else:
            ds_new = xr.Dataset(coords=ds_src.coords,\
                                attrs=ds_src.attrs,\
                                )
        ds_new = ds_new.drop_dims('valid_time')
        
        src_time = ds_src.valid_time.values - np.timedelta64(30,'m')
        first_day = src_time[12].astype('datetime64[D]')
        last_day = src_time[-12].astype('datetime64[D]')
        last_day_p1 = last_day + np.timedelta64(1,'D')
        new_time = np.arange(first_day,last_day_p1,np.timedelta64(1,'D'))\
                        .astype('datetime64[ns]') + np.timedelta64(12,'h')

        time_axis_num = dict()
        new_datavars = dict()
        for varname in data_var_names:
            data_sizes = ds_src[varname].sizes
            if 'expver' in data_sizes:
                data_sizes = dict(data_sizes)
                del data_sizes['expver']
            dim_tuple = tuple()
            for dim_num,curr_dim in enumerate(data_sizes):
                if curr_dim == 'valid_time':
                    time_axis_num = {**time_axis_num,**{varname:dim_num}}
                    dim_tuple += (len(new_time),)
                else:
                    dim_tuple += (data_sizes[curr_dim],)
            curr_array = np.empty(dim_tuple).astype('float32')
            curr_array.fill(np.nan)
            new_datavars = {**new_datavars,**{varname:[tuple(data_sizes),\
                                                       curr_array,\
                                                       ds_src[varname].attrs]}}

        ds_src.close()
        
        # if __name__ == '__main__':
            # pool = multiprocessing.Pool(processes=3)
            # vars_avg_all = pool.map(daily_avg,zip(new_time,\
            #                                        [source_file]*len(new_time)))
        vars_avg_all = []
        for new_t in new_time:
            vars_avg_all.append(daily_avg([new_t.astype('datetime64[D]'),source_file]))
            # pool.close()
            # pool.join()
            
        for curr_ind,vars_avg in enumerate(vars_avg_all):    
            # write averages to data variable dictionary
            for varname in data_var_names:
                curr_array = new_datavars[varname][1]
                if time_axis_num[varname] == 0:
                    index_str = str(curr_ind)+',...'
                else:
                    index_str = (':,'*(time_axis_num[varname]))+str(curr_ind)+',...'
                exec('curr_array['+index_str+'] = vars_avg[varname]')
                new_datavars[varname][1] = curr_array

        # write daily-averaged fields to new dataset
        ds_new = ds_new.assign_coords({'valid_time':new_time})
        for varname in data_var_names:
            ds_new[varname] = tuple(new_datavars[varname])
        
        del new_datavars

        
        # write new dataset to output file
        
        # ds_new.to_netcdf(path=output_filepathform+year_str+'_'+month_str+'.nc',format="NETCDF4")
        ds_new.to_netcdf(path=output_filepathform+year_str+'-'+month_str+'.nc',format="NETCDF4")
        print('Created daily mean file for year '+year_str+', month '+month_str)

        del ds_new