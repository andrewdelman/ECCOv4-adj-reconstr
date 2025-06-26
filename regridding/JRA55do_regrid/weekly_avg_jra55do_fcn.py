def weekly_avg(inputs):    
    import numpy as np
    import xarray as xr
    
    bin_start,bin_end,bin_center,source_filepathform = inputs
    
    bin_start_year = bin_start.astype('datetime64[Y]')
    bin_end_year = bin_end.astype('datetime64[Y]')
    
    ds_src = xr.open_mfdataset(source_filepathform\
                                    +str(bin_start_year)+'*.nc')
    data_var_names = list(ds_src.keys())
    for varname in list(data_var_names):
        if 'bnds' in varname:
            data_var_names.remove(varname)
    
    if bin_end_year - bin_start_year > np.timedelta64(0,'Y'):
        for bin_year in np.arange(bin_start_year+np.timedelta64(1,'Y'),\
                                   bin_end_year+np.timedelta64(1,'Y'),\
                                   np.timedelta64(1,'Y')):
            ds_next = xr.open_mfdataset(source_filepathform\
                                            +str(bin_year)+'*.nc')
            ds_src = xr.concat((ds_src,ds_next),dim='time',\
                               compat='override',coords='minimal',data_vars='minimal')
    
    src_time = ds_src.time.values
    in_bin = np.logical_and(src_time >= bin_start,src_time < bin_end).nonzero()[0]
    ds_in_bin = ds_src.isel(time=in_bin)
    curr_var_avgs = dict()
    for varname in data_var_names:
        curr_array_avg = ds_in_bin[varname].mean(dim="time",keepdims=True).values
        curr_var_avgs = {**curr_var_avgs,**{varname:curr_array_avg}}
    
    return curr_var_avgs
