def daily_avg(inputs):
    
    import numpy as np
    import xarray as xr
    
    day,source_file = inputs
    
    source_filepathform = source_file[:-10]
    year = int(source_file[-10:-6])
    year_str = str(year)
    month_str = source_file[-5:-3]
    
    ds_src = xr.open_mfdataset(source_file)
    src_time = ds_src.valid_time.values - np.timedelta64(30,'m')
    
    first_day = src_time[12].astype('datetime64[D]')
    last_day = src_time[-12].astype('datetime64[D]')
    last_day_p1 = last_day + np.timedelta64(1,'D')
    
    if last_day_p1.astype('datetime64[M]') - last_day.astype('datetime64[M]') > np.timedelta64(0,'M'):
        if month_str == '12':
            end_source_file = source_filepathform+str(year+1)+'-01.nc'
        else:
            end_month_str = str(int(month_str)+1).rjust(2,'0')
            end_source_file = source_filepathform+year_str+'-'+end_month_str+'.nc'
        ds_src_end = xr.open_mfdataset(end_source_file)
        ds_src_end = ds_src_end.isel(valid_time=[0])
        if (('expver' in ds_src.dims) and ('expver' not in ds_src_end.dims)):
            ds_src_end = ds_src_end.assign_coords({'expver':ds_src.expver[-1]})
            for varname in list(ds_src_end.keys()):
                ds_src_end[varname] = ds_src_end[varname].expand_dims(dim='expver',axis=1)
            ds_src = xr.concat((ds_src,ds_src_end),dim='valid_time',\
                           data_vars='minimal',coords='minimal',compat='broadcast_equals')
        elif (('expver' not in ds_src.dims) and ('expver' in ds_src_end.dims)):
            ds_src = ds_src.assign_coords({'expver':ds_src_end.expver[0]})
            for varname in list(ds_src.keys()):
                ds_src[varname] = ds_src[varname].expand_dims(dim='expver',axis=1)
            ds_src = xr.concat((ds_src,ds_src_end),dim='valid_time',\
                           data_vars='minimal',coords='minimal',compat='broadcast_equals')
        else:
            ds_src = xr.concat((ds_src,ds_src_end),dim='valid_time',\
                           data_vars='minimal',coords='minimal',compat='override')
        src_time = ds_src.valid_time.values - np.timedelta64(30,'m')
        del ds_src_end
    
    
    # data variable names
    data_var_names = list(ds_src.keys())
    
    # identify hours in the current day
    in_day = np.logical_and(src_time - np.datetime64(str(day)+'T00:00:00','ns')\
                             >= np.timedelta64(0,'ns'),\
                             src_time - np.datetime64(str(day+np.timedelta64(1,'D'))+'T00:00:00','ns')\
                             < np.timedelta64(0,'ns')).nonzero()[0]
    ds_src_inday = ds_src.isel(valid_time=in_day)
    if 'expver' in ds_src_inday.dims:
        curr_var = ds_src_inday[list(ds_src_inday.data_vars)[0]]
        n_nans = np.sum(np.sum(~np.isnan(curr_var.values),axis=-2),axis=-1)
        for n_dim,dim_name in enumerate(curr_var.dims):
            if dim_name == 'expver':
                exp_dimnum = n_dim
        which_ver = np.argmax(n_nans,axis=exp_dimnum)
        ver_0_time = (which_ver == 0).nonzero()[0]
        ver_1_time = (which_ver == 1).nonzero()[0]

    new_avgs = dict()
    for varname in data_var_names:
        if 'expver' in ds_src_inday[varname].dims:
            curr_var = ds_src_inday[varname]
            curr_var_concat = np.concatenate((curr_var[ver_0_time,0,:,:],\
                                              curr_var[ver_1_time,1,:,:]),\
                                              axis=0)
            src_array_daymean = np.nanmean(curr_var_concat,axis=0,keepdims=True)
        else:
            src_array_daymean = ds_src_inday[varname].mean(dim="valid_time",keepdims=True).values
        new_avgs = {**new_avgs,**{varname:src_array_daymean}}

    del ds_src_inday
    ds_src.close()

    
    print('Completed day = '+str(day))
    
    return new_avgs