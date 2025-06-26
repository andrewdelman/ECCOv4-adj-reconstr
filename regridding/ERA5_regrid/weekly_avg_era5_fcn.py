def weekly_avg(inputs):    
    import numpy as np
    import xarray as xr
    
    bin_start,bin_end,bin_center,source_filepathform = inputs
    
    bin_start_month = (bin_start + np.timedelta64(12,'h')).astype('datetime64[M]')
    bin_end_month = (bin_end - np.timedelta64(12,'h')).astype('datetime64[M]')
    
    ds_src = xr.open_mfdataset(source_filepathform\
                                    +str(bin_start_month)[-7:-3]+'_'+str(bin_start_month)[-2:]+'.nc')
    data_var_names = list(ds_src.keys())
    
    if bin_end_month - bin_start_month > np.timedelta64(0,'M'):
        for bin_month in np.arange(bin_start_month+np.timedelta64(1,'M'),\
                                   bin_end_month+np.timedelta64(1,'M'),\
                                   np.timedelta64(1,'M')):
            ds_next = xr.open_mfdataset(source_filepathform\
                                            +str(bin_month)[-7:-3]+'_'+str(bin_month)[-2:]+'.nc')
            ds_src = xr.concat((ds_src,ds_next),dim='time',\
                               compat='override',coords='minimal',data_vars='minimal')
    
    src_time = ds_src.time.values
    src_time += np.timedelta64(12,'h')
    in_bin = np.logical_and(src_time >= bin_start,src_time < bin_end).nonzero()[0]
    ds_in_bin = ds_src.isel(time=in_bin)
    curr_var_avgs = dict()
    for varname in data_var_names:
        curr_array_avg = ds_in_bin[varname].mean(dim="time",keepdims=True).values
        curr_var_avgs = {**curr_var_avgs,**{varname:curr_array_avg}}
    
    return curr_var_avgs