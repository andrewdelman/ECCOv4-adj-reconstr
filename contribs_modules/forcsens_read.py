#!/usr/bin/env python
# coding: utf-8



def nanmask_create(array):
    import numpy as np
    
    mask = np.logical_and(~np.isnan(array),~np.isinf(array))\
      .astype('int64')
    return mask


def trend_compute_dimvec(input_array,dim_vec,axis=-1,output_trendline=True):
    """
    Computes linear regression-based trend in input_array, along specified axis with axis coordinates in dim_vec.
    Outputs trend values in reg_trend.
    If output_trendline = True, also outputs trend lines (that can be subtracted from input_array) in
    reg_trendline."""
    
    pass
    
    import numpy as np
    
    input_reshaped = np.moveaxis(input_array,axis,0)
    input_shape = input_reshaped.shape
    mask = np.asarray(np.logical_and(np.logical_and(~np.isnan(input_reshaped),~np.isinf(input_reshaped)),\
                                      np.abs(input_reshaped) > 1e-10*np.nanmax(np.abs(input_reshaped))))
    dim_vec_tiled = np.moveaxis(np.tile(dim_vec,input_shape[1:]+(1,)),-1,0)
    dim_vec_nomean = dim_vec_tiled - (np.sum(mask*dim_vec_tiled,axis=0)/np.sum(mask,axis=0))
    input_mean = np.nansum(mask*input_reshaped,axis=0,keepdims=True)/np.nansum(mask,axis=0,keepdims=True)
    input_mean = np.where(~np.isnan(input_mean),input_mean,0)
    input_nomean = input_reshaped - input_mean
    reg_num = np.nansum(mask*(dim_vec_nomean*input_nomean),axis=0,keepdims=True)
    reg_denom = np.nansum(mask*(dim_vec_nomean**2),axis=0,keepdims=True)
    reg_trend = (np.moveaxis(reg_num/reg_denom,0,axis))
    if output_trendline == True:
        reg_trendline = np.moveaxis(input_mean,0,axis) + (reg_trend*np.moveaxis(dim_vec_nomean,0,axis))
        return reg_trend,reg_trendline
    else:    
        return reg_trend


def var_compute(array,axis=-1,keepdims=False):
    import numpy as np
    
    mask = nanmask_create(array)
    curr_mean = np.nansum(mask*array,axis=axis,keepdims=True)/np.nansum(mask,axis=axis,keepdims=True)
    curr_var = np.nansum(mask*((array - curr_mean)**2),axis=axis,keepdims=keepdims)/\
                 np.nansum(mask,axis=axis,keepdims=keepdims)
    
    return curr_var



def data_llcshape(data):
    import numpy as np
    
    "Shape output into 13 tiles configuration"
    len_j_i = 90    # length of each side of tile
    n_nonspat = int(data.size/(13*(len_j_i**2)))
    
    data_shaped = np.reshape(data,(n_nonspat,13,len_j_i,len_j_i))
    ind_tile7start = np.reshape(np.arange(7*(len_j_i**2),data.size,13*(len_j_i**2)),(-1,1))
    ind_tile10start = np.reshape(np.arange(10*(len_j_i**2),data.size,13*(len_j_i**2)),(-1,1))
    ind_tile7to9 = (ind_tile7start + np.arange(0,3*(len_j_i**2))).flatten()
    ind_tile10to12 = (ind_tile10start + np.arange(0,3*(len_j_i**2))).flatten()
    data_shaped[:,7:10,:,:] = np.moveaxis(np.reshape(data[ind_tile7to9],(n_nonspat,len_j_i,3,len_j_i)),-2,-3)
    data_shaped[:,10:13,:,:] = np.moveaxis(np.reshape(data[ind_tile10to12],(n_nonspat,len_j_i,3,len_j_i)),-2,-3)
    
    return data_shaped



def preprocess_pred_time(ds):
    "Preprocess prediction time in arrays for which new init_time dimension is being added (e.g., CESM)."
    
    ds = ds.assign_coords({'pred_time':ds['time'].copy().expand_dims(dim='init_time',axis=0),\
                           'pred_time_bound':ds['time_bound'].copy()\
                                                       .expand_dims(dim='init_time',axis=0)})
    del ds['time']
    del ds['time_bound']

    return ds


def pred_endmo_noleap_to_midmo_withleap(ds_curr):
    import numpy as np
    
    # convert prediction times from end of month no leap to mid-month with leap days
    pred_times = np.datetime64('1977-01-01','ns')\
                  + (((ds_curr.pred_time.values\
                  - 721605)/365)*365.24*8.64e13).astype('timedelta64[ns]')
    pred_times -= np.timedelta64(int(15*8.64e13),'ns')
    possible_pred_time_bounds = np.arange(np.datetime64('1964-01','M'),np.datetime64('2040-01','M'),\
                                          np.timedelta64(1,'M')).astype('datetime64[ns]')
    possible_pred_times = possible_pred_time_bounds[:-1] + (np.diff(possible_pred_time_bounds)/2)
    for (init_ind,time_ind),old_pred_time in np.ndenumerate(pred_times):
        closest_ind = np.nanargmin(np.abs(possible_pred_times - old_pred_time))
        if np.abs(possible_pred_times[closest_ind] - old_pred_time) > np.timedelta64(3,'D'):
            raise ValueError('No match to this prediction month found.\n'\
                             +'May need to expand the range of possible_pred_time_bounds.')
        pred_times[init_ind,time_ind] = possible_pred_times[closest_ind]

    return pred_times




def adj_sens_read(curr_forc,adj_sens_dir,adj_sens_loc,pred_month):
    import numpy as np
    from os.path import join,expanduser    
    
    "Read in adjoint sensitivities for a given forcing, prediction location, and prediction month (of the year)"
    
    curr_filename = join(adj_sens_dir,adj_sens_loc,\
                          (str(pred_month).rjust(2,'0'))+'04','adxx_'+curr_forc+'.0000000129_dim.data')
    data = np.fromfile(curr_filename,dtype='>f4')
    
    curr_sens = data_llcshape(data)
    
    # smooth sensitivities after first timestep
    curr_sens_smoothed = curr_sens.copy()
    for tstep in range(1,curr_sens.shape[0]):
        curr_sens_smoothed[tstep,:,:,:] = ((4/7)*curr_sens[tstep-1,:,:,:]) + ((3/7)*curr_sens[tstep,:,:,:])
    
    adj_tval_final = {'1':np.datetime64('2004-02-04','ns'),\
                  '2':np.datetime64('2004-03-03','ns'),\
                  '3':np.datetime64('2004-04-07','ns'),\
                  '4':np.datetime64('2004-05-05','ns'),\
                  '5':np.datetime64('2004-06-02','ns'),\
                  '6':np.datetime64('2004-07-07','ns'),\
                  '7':np.datetime64('2004-08-04','ns'),\
                  '8':np.datetime64('2004-09-08','ns'),\
                  '9':np.datetime64('2004-10-06','ns'),\
                  '10':np.datetime64('2004-11-03','ns'),\
                  '11':np.datetime64('2004-12-08','ns'),\
                  '12':np.datetime64('2005-01-05','ns')}

    sens_tval_tsteps = adj_tval_final[str(pred_month)] - np.timedelta64(int(3.5*8.64e13),'ns')\
                    + (np.arange(-curr_sens.shape[0]+1,1)*np.timedelta64(int(7*8.64e13),'ns'))
    
    return curr_sens_smoothed,sens_tval_tsteps



def adj_sens_read_multiyr(curr_forc,adj_sens_dir,adj_sens_loc):
    import numpy as np
    from os.path import join,expanduser    
        
    "Read in multi-year (1992-2015) adjoint sensitivities for a given forcing and prediction location"
    
    curr_filename = join(adj_sens_dir,adj_sens_loc,\
                          '1992_2015','adxx_'+curr_forc+'.0000000059_dim.data')
    data = np.fromfile(curr_filename,dtype='>f4')
    
    curr_sens = data_llcshape(data)
    
    # smooth sensitivities after first timestep
    curr_sens_smoothed = curr_sens.copy()
    for tstep in range(1,curr_sens.shape[0]):
        curr_sens_smoothed[tstep,:,:,:] = ((4/7)*curr_sens[tstep-1,:,:,:]) + ((3/7)*curr_sens[tstep,:,:,:])
    
    adj_tval_final = np.datetime64('2016-01-06','ns')
    sens_tval_tsteps = adj_tval_final - np.timedelta64(int(3.5*8.64e13),'ns')\
                    + (np.arange(-curr_sens.shape[0]+1,1)*np.timedelta64(int(7*8.64e13),'ns'))
    
    return curr_sens_smoothed,sens_tval_tsteps




def ecco_forc_read_weekly(curr_forc,forc_dir,years_to_predict,max_weeks_lead,closest_forc_to_sens_atyears):
    import numpy as np
    import xarray as xr
    from os.path import join
    
    closest_forc_to_sens_atyears_unique = np.unique(closest_forc_to_sens_atyears)
    in_time_range = np.logical_and(closest_forc_to_sens_atyears_unique\
                                   - np.datetime64(str(years_to_predict[0])+'-01-01','ns') >= -np.timedelta64(7*max_weeks_lead,'D'),\
                                   closest_forc_to_sens_atyears_unique\
                                   - np.datetime64(str(years_to_predict[-1]+1)+'-01-10','ns')\
                                   < np.timedelta64(0,'ns')).nonzero()[0]
    
    closest_forc_to_sens_atyears_unique = closest_forc_to_sens_atyears_unique[in_time_range]
    closest_forc_hr_to_sens_tstep_unique = \
                        np.round(((closest_forc_to_sens_atyears_unique - np.datetime64('1992-01-01','ns'))\
                        .astype('float64'))/(3.6e12)).astype('int64')\
                        + 84
    
    forc_file_str_1 = curr_forc+'_weekly_v1.'
    forc_file_str_2 = '.data'
    
    curr_forc_array = np.empty((len(closest_forc_hr_to_sens_tstep_unique),13,90,90))
    curr_forc_array.fill(np.nan)
    for curr_tstep,forc_hr in enumerate(closest_forc_hr_to_sens_tstep_unique):
        curr_filename = join(forc_dir,forc_file_str_1+str(forc_hr).rjust(10,'0')+forc_file_str_2)
        try:
            data = np.fromfile(curr_filename,dtype='>f4')
        except:
            continue
        curr_forc_array[curr_tstep,:,:,:] = data_llcshape(data).squeeze()
    
    curr_da = xr.DataArray(\
                           data=curr_forc_array,\
                           dims=['time','tile','j','i'],\
                           coords={'time':closest_forc_to_sens_atyears_unique},\
                           )
    
    return curr_da



def ecco_emp_forc_read_weekly(forc_dir,years_to_predict,max_weeks_lead):
    import numpy as np
    import xarray as xr
    from os.path import join
    
    ds_ecco_epr = xr.open_mfdataset(join(forc_dir,'*.zarr'),engine='zarr',\
        compat='override',data_vars='minimal',coords='minimal')
    curr_da = 1000*(ds_ecco_epr.EXFevap - ds_ecco_epr.EXFpreci)

    in_time_range = np.logical_and(curr_da.time.values - np.datetime64(str(years_to_predict[0])+'-01-01','ns')\
                                   >= -np.timedelta64(7*max_weeks_lead,'D'),\
                                   curr_da.time.values - np.datetime64(str(years_to_predict[-1]+1)+'-01-10','ns')\
                                   < np.timedelta64(0,'ns')).nonzero()[0]
    curr_da = curr_da.isel(time=in_time_range).compute()

    return curr_da



def era5_forc_read_weekly(curr_forc,forc_dir,years_to_predict,max_weeks_lead):
    import numpy as np
    import xarray as xr
    from os.path import join
    
    ds_forc_all = xr.open_mfdataset(join(forc_dir,'*.nc'),\
                                compat='override',data_vars='minimal',coords='minimal')

    if curr_forc == 'empmr':
        curr_da = -ds_forc_all.mer - ds_forc_all.mtpr - ds_forc_all.mror
    elif curr_forc == 'emp':
        curr_da = -ds_forc_all.mer - ds_forc_all.mtpr
    elif curr_forc == 'qnet':
        curr_da = -(ds_forc_all.msnswrf + ds_forc_all.msnlwrf + ds_forc_all.mslhf + ds_forc_all.msshf)
    elif curr_forc == 'tauu':
        curr_da = -(ds_forc_all.metss + ds_forc_all.megwss)
    elif curr_forc == 'tauv':
        curr_da = -(ds_forc_all.mntss + ds_forc_all.mngwss)
    
    in_time_range = np.logical_and(ds_forc_all.time.values - np.datetime64(str(years_to_predict[0])+'-01-01','ns')\
                                   >= -np.timedelta64(7*max_weeks_lead,'D'),\
                                   ds_forc_all.time.values - np.datetime64(str(years_to_predict[-1]+1)+'-01-10','ns')\
                                   < np.timedelta64(0,'ns')).nonzero()[0]
    curr_da = curr_da.isel(time=in_time_range).compute()
    
    return curr_da



def jra55do_friver_forc_read_weekly(forc_dir,years_to_predict,max_weeks_lead):
    import numpy as np
    import xarray as xr
    from os.path import join
    
    ds_forc_all = xr.open_mfdataset(join(forc_dir,'*.nc'),\
                                compat='override',data_vars='minimal',coords='minimal')

    curr_da = -ds_forc_all.friver
    
    in_time_range = np.logical_and(ds_forc_all.time.values - np.datetime64(str(years_to_predict[0])+'-01-01','ns')\
                                   >= -np.timedelta64(7*max_weeks_lead,'D'),\
                                   ds_forc_all.time.values - np.datetime64(str(years_to_predict[-1]+1)+'-01-10','ns')\
                                   < np.timedelta64(0,'ns')).nonzero()[0]
    curr_da = curr_da.isel(time=in_time_range).compute()
    
    return curr_da





def seas5_ensmean_forc_read_weekly(curr_forc,forc_dir,years_to_predict,max_weeks_lead):
    import numpy as np
    import xarray as xr
    from os.path import join
    
    ds_forc_all = xr.open_mfdataset(join(forc_dir,'*meanstd*'),\
                                    engine='zarr',\
                                    combine='nested',\
                                    concat_dim='init_time',\
                                    compat='override',\
                                    data_vars='minimal',coords='minimal')

    if curr_forc == 'empmr':
        curr_da = -1000*(ds_forc_all.e_ens_mean + ds_forc_all.tp_ens_mean \
                   + ds_forc_all.ro_ens_mean.where(\
                   ~np.isnan(ds_forc_all.ro_ens_mean),0))
    elif curr_forc == 'qnet':
        curr_da = -(ds_forc_all.ssr_ens_mean + ds_forc_all.str_ens_mean \
                    + ds_forc_all.slhf_ens_mean + ds_forc_all.sshf_ens_mean)
    elif curr_forc == 'tauu':
        curr_da = -ds_forc_all.ewss_ens_mean
    elif curr_forc == 'tauv':
        curr_da = -ds_forc_all.nsss_ens_mean
    
    if curr_forc != 'msl':
        pred_minus_init = (ds_forc_all.pred_time - ds_forc_all.init_time).values
        week_time_span = np.where(pred_minus_init < np.timedelta64(7,'D'),\
                                  pred_minus_init.astype('timedelta64[s]')\
                                    .astype('float64') + (3.5*86400),\
                                  7*86400)
        curr_da = curr_da/np.expand_dims(week_time_span,axis=(-1,-2,-3))
    
    in_init_time_range = np.unique(\
               np.logical_and(ds_forc_all.pred_time.values - np.datetime64(str(years_to_predict[0])+'-01-01','ns')\
                                   >= -np.timedelta64(7*max_weeks_lead,'D'),\
                                   ds_forc_all.pred_time.values - np.datetime64(str(years_to_predict[-1]+1)+'-01-10','ns')\
                                   < np.timedelta64(0,'ns')).nonzero()[0])
    curr_da = curr_da.isel(init_time=in_init_time_range)
    
    return curr_da



def cesm_dple_ensmean_forc_read_monthly(curr_forc,forc_dir,years_to_predict,max_weeks_lead):
    import numpy as np
    import xarray as xr
    from os.path import join
    import glob
    
    if curr_forc == 'empmr':
        cesm_var_id = 'SFWF'
    elif curr_forc == 'qnet':
        cesm_var_id = 'SHF'
    elif curr_forc == 'tauu':
        cesm_var_id = 'TAUU'
    elif curr_forc == 'tauv':
        cesm_var_id = 'TAUV'
    
    file_list = glob.glob(join(forc_dir,cesm_var_id,'*.nc'))
    file_list_sorted = sorted(file_list)
    
    ds_forc_all = xr.open_mfdataset(file_list_sorted,\
                                    combine='nested',\
                                    concat_dim='init_time',\
                                    preprocess=preprocess_pred_time,\
                                    data_vars=[cesm_var_id],coords=['pred_time','pred_time_bound'],\
                                    decode_times=False)
    
    pred_times = pred_endmo_noleap_to_midmo_withleap(ds_forc_all)

    ds_forc_all = ds_forc_all.assign_coords({'pred_time':\
                            (['init_time','time'],pred_times)})
    del ds_forc_all['pred_time_bound']
    
    init_time = np.empty((0,),dtype='datetime64[ns]')
    for file_count,filename in enumerate(file_list_sorted):
        curr_init_yrmo = filename.split('.')[-2][:6]
        curr_init_time = np.datetime64(str(curr_init_yrmo)[0:4]+'-'\
                           +str(curr_init_yrmo)[4:6]+'-01','ns')
        init_time = np.append(init_time,curr_init_time)
    ds_forc_all = ds_forc_all.assign_coords({'init_time':\
                            (['init_time'],init_time)})

    if curr_forc == 'empmr':
        curr_da = -ds_forc_all.SFWF
    elif curr_forc == 'qnet':
        curr_da = -ds_forc_all.SHF
    elif curr_forc == 'tauu':
        curr_da = -0.1*ds_forc_all.TAUU
    elif curr_forc == 'tauv':
        curr_da = -0.1*ds_forc_all.TAUV
    
    in_init_time_range = np.unique(\
             np.logical_and(ds_forc_all.pred_time.values - np.datetime64(str(years_to_predict[0])+'-01-01','ns')\
                                   >= -np.timedelta64(7*max_weeks_lead,'D'),\
                                   ds_forc_all.pred_time.values - np.datetime64(str(years_to_predict[-1]+1)+'-01-10','ns')\
                                   < np.timedelta64(0,'ns')).nonzero()[0])
    curr_da = curr_da.isel(init_time=in_init_time_range)
    
    return curr_da



def cesm_hrdp_ensmean_forc_read_monthly(curr_forc,forc_dir,years_to_predict,max_weeks_lead):
    import numpy as np
    import xarray as xr
    from os.path import join
    import glob
    
    if curr_forc == 'empmr':
        cesm_var_id = 'SFWF'
    elif curr_forc == 'qnet':
        cesm_var_id = 'SHF'
    elif curr_forc == 'tauu':
        cesm_var_id = 'TAUU'
    elif curr_forc == 'tauv':
        cesm_var_id = 'TAUV'
    
    file_list = glob.glob(join(forc_dir,cesm_var_id,'*.zarr'))
    file_list_sorted = sorted(file_list)
    
    ds_forc_all = xr.open_mfdataset(file_list_sorted,\
                                    engine='zarr',\
                                    combine='nested',\
                                    concat_dim='init_time',\
                                    preprocess=preprocess_pred_time,\
                                    data_vars=[cesm_var_id],coords=['pred_time','pred_time_bound'],\
                                    decode_times=False)
    
    pred_times = pred_endmo_noleap_to_midmo_withleap(ds_forc_all)

    ds_forc_all = ds_forc_all.assign_coords({'pred_time':\
                            (['init_time','time'],pred_times)})
    del ds_forc_all['pred_time_bound']
    
    init_time = np.empty((0,),dtype='datetime64[ns]')
    for file_count,filename in enumerate(file_list_sorted):
        curr_init_yrmo = filename.split('.')[-4]
        curr_init_time = np.datetime64(str(curr_init_yrmo)[0:4]+'-'\
                           +str(curr_init_yrmo)[5:7]+'-01','ns')
        init_time = np.append(init_time,curr_init_time)
    ds_forc_all = ds_forc_all.assign_coords({'init_time':\
                            (['init_time'],init_time)})

    if curr_forc == 'empmr':
        curr_da = -ds_forc_all.SFWF.mean('ens_mem')
    elif curr_forc == 'qnet':
        curr_da = -ds_forc_all.SHF.mean('ens_mem')
    elif curr_forc == 'tauu':
        curr_da = -0.1*ds_forc_all.TAUU.mean('ens_mem')
    elif curr_forc == 'tauv':
        curr_da = -0.1*ds_forc_all.TAUV.mean('ens_mem')
    
    in_init_time_range = np.unique(\
              np.logical_and(ds_forc_all.pred_time.values - np.datetime64(str(years_to_predict[0])+'-01-01','ns')\
                                   >= -np.timedelta64(7*max_weeks_lead,'D'),\
                                   ds_forc_all.pred_time.values - np.datetime64(str(years_to_predict[-1]+1)+'-01-10','ns')\
                                   < np.timedelta64(0,'ns')).nonzero()[0])
    curr_da = curr_da.isel(init_time=in_init_time_range)
    
    return curr_da