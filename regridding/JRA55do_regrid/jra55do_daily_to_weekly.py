# create weekly mean JRA55-do files (spanning a year)
# from daily mean JRA55-do files (spanning a year)

import numpy as np
import xarray as xr
from os.path import expanduser,join
import sys

import multiprocessing
sys.path.append(join('home5','adelman'))
from weekly_avg_jra55do_fcn import *


## Path and fileform (start of filenames) for daily JRA55-do data
source_filepath = join('/nobackup','adelman','JRA55-do','friver','daily')
source_fileform = 'friver_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-6-0_gr_'
source_filepathform = join(source_filepath,source_fileform)

## Path and fileform for output weekly JRA55-do data
output_filepath = join('/nobackup','adelman','JRA55-do','friver','weekly')
output_fileform = 'friver_weekly-JRA55-do-1-6-0_gr_'

output_filepathform = join(output_filepath,output_fileform)

## Year range to process (unlike usual Python convention, end_year is included in the range)
start_year = 1980
end_year = 1999


# weekly bins that agree with the weekly averages of ECCO forcing
weekly_bin_starts = np.arange(np.datetime64('1964-01-01','ns'),np.datetime64('2030-01-01','ns'),\
                              np.timedelta64(int(7*8.64e13),'ns'))
weekly_bin_ends = weekly_bin_starts + np.timedelta64(7,'D')
weekly_bin_centers = weekly_bin_starts + np.timedelta64(int(3.5*24),'h')

bins_in_range = np.logical_and(weekly_bin_centers >= np.datetime64(str(start_year)+'-01-01','ns'),\
                                 weekly_bin_centers < np.datetime64(str(end_year+1)+'-01-01','ns')).nonzero()[0]
weekly_bin_starts = weekly_bin_starts[bins_in_range]
weekly_bin_ends = weekly_bin_ends[bins_in_range]
weekly_bin_centers = weekly_bin_centers[bins_in_range]

ds_src = xr.open_mfdataset(source_filepathform+str(start_year)+'*.nc')
data_var_names = list(ds_src.keys())

for curr_year in range(start_year,end_year+1):
    bins_in_range = np.logical_and(weekly_bin_centers >= np.datetime64(str(curr_year)+'-01-01','ns'),\
                                 weekly_bin_centers < np.datetime64(str(curr_year+1)+'-01-01','ns')).nonzero()[0]
    curr_bin_starts = weekly_bin_starts[bins_in_range]
    curr_bin_ends = weekly_bin_ends[bins_in_range]
    curr_bin_centers = weekly_bin_centers[bins_in_range]
    
    ds_new = xr.Dataset(coords=ds_src.coords,\
                    attrs=ds_src.attrs,\
                    )
    new_time = curr_bin_centers
    time_axis_num = dict()
    new_datavars = dict()
    for varname in list(data_var_names):
        data_sizes = ds_src[varname].sizes
        dim_tuple = tuple()
        for dim_num,curr_dim in enumerate(data_sizes):
            if curr_dim == 'time':
                time_axis_num = {**time_axis_num,**{varname:dim_num}}
                dim_tuple += (len(new_time),)
            else:
                dim_tuple += (data_sizes[curr_dim],)
        if 'bnds' in varname:
            new_datavars = {**new_datavars,**{varname:ds_src[varname]}}
            data_var_names.remove(varname)
        else:
            curr_array = np.empty(dim_tuple).astype('float32')
            curr_array.fill(np.nan)
            new_datavars = {**new_datavars,**{varname:[ds_src[varname].dims,\
                                                       curr_array,\
                                                       ds_src[varname].attrs]}}

    ds_new['time'] = new_time
    time_bounds = np.hstack((np.reshape(curr_bin_starts,(-1,1)),\
                             np.reshape(curr_bin_ends,(-1,1))))
    ds_new = ds_new.assign_coords(time_bounds=(['time','bound'],\
                                               time_bounds))
    ds_new['time_bounds'] = ds_new['time_bounds'].assign_attrs(\
                                    long_name='Time bounds of weekly mean')                                               

    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=3)
        vars_avg_all = pool.map(weekly_avg,\
                                   zip(curr_bin_starts,curr_bin_ends,curr_bin_centers,\
                                    [source_filepathform]*len(curr_bin_starts)))
        pool.close()
        pool.join()
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
        print('Added bin centered on '+str(curr_bin_centers[curr_ind].astype('datetime64[D]')))
    
    
    # write weekly-averaged fields to new dataset
    for varname in data_var_names:
        ds_new[varname] = tuple(new_datavars[varname])
    
    del new_datavars
    
    # write weekly-averaged dataset to output file
    ds_new.to_netcdf(path=output_filepathform+str(curr_year)+'.nc',format="NETCDF4")
    print('Created weekly mean file for year '+str(curr_year))
    
    del ds_new
