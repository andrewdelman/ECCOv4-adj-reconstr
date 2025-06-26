#!/usr/bin/env python
# coding: utf-8

from forcsens_read import *
from numba import jit
    
def forcsens_var_cells(inputs):
    """Computes residual variance or variance associated with forcing*sensitivity
       (i.e., impact), from each cell as indicated by the cell_indices dictionary.
       Takes as input a dictionary (inputs) consisting of the following
       entries:
       curr_forc: str, forcing name to compute impact variances for
       forcing_preinit: dict consisting of
           forc_type: str, type of forcing (e.g., 'era5', 'ecco','era5_sensadj')
           forc_dir: str, directory where forc_type is located
           forc_runoff_type: str, type/source of coastal runoff/river discharge (e.g., 'jra55do')
           forc_runoff_dir = str, directory where forc_runoff_type is located
           forc_subtract_type: (optional) str, type of forcing to be subtracted from forc_type
           forc_subtract_dir: (optional) str, directory where forc_subtract_type is located
       forcing_postinit: (optional) dict consisting of
           forc_type: str, type of predicted forcing (e.g., 'ecmwf', 'eccc')
           forc_dir: str, directory where forc_type is located
           forc_subtract_type: (optional) str, type of forcing to be subtracted from forc_type
           forc_subtract_dir: (optional) str, directory where forc_subtract_type is located       
       obs, reconstr, or model: (optional) dict consisting of
           tseries: 1-D numpy float array, observation or reconstruction time series to use for
                    residual variance calculations
           datetime: 1-D numpy datetime64 array, times corresponding to tseries data
       **Note: if a reconstr or obs dictionary is given in inputs, residual variances are computed.
         Otherwise, the variances of the impacts are computed.
       sensadj_params: (optional, only if a sensadj forcing is used) dict consisting of
           base_pred_file: str, file to use for base prediction (provides reconstr time series and stddev distribution)
           obs: dict, consisting of
               tseries: 1-D numpy float array, observation time series to use for
                        reducing the residual (vs the reconstr time series)
               datetime: 1-D numpy datetime64 array, times corresponding to tseries data
           obs_weights_array: 3-D numpy float array, weights for partitioning the obs tseries, 
                              must sum to 1 for each prediction month
           years_to_use: 1-D numpy int array, years to use when computing sensitivity adjustment ratios
           trend_remove: (optional) bool, if True then secular trend is removed for sensadj computation
           n_eofs: int, maximum number of EOFs used to compute sensitivity adjustment ratios
           ratio_limits: list or 2-element float array, sets the min and max respectively of allowed adjustment ratios
       adj_sens_loc: str, location name for adjoint sensitivities (e.g., 'Pensacola', 'EasternGulfCoast')
       adj_sens_dir_seas: str, directory where seasonally-varying, < 12-month lead time adjoint sensitivities are located
       adj_sens_dir_multiyr: str, directory where multi-year (>= 12-month lead time) adjoint sensitivities are located
       pred_loc: str, location name for prediction site. 
                 Only needs to be specified if sensadj is used, this may or may not be the same as adj_sens_loc.
       years_to_predict: 1-D numpy int array, years to include in assessment
       trend_remove: (optional) bool, if True then secular trend is removed for variance computation
       max_weeks_lead: int, maximum lead time (in weeks) to include forcing for
       multiyr_seas_taper_weeks: int, number of weeks (starting at 12 months and progressing to shorter lead times) 
                                 where multi-year and seasonal adjoint sensitivities are tapered together.
       sens_time: 2-D numpy datetime64 array, dates of seasonal adjoint sensitivity time steps
       sens_time_multiyr: 2-D numpy datetime64 array, dates of multi-year sensitivity time steps
       closest_forc_to_sens_atyears: 3-D numpy datetime64 array, forcing times corresponding to 
                                     each forcing month, sensitivity time step, and year
       cell_indices: list consisting of dictionaries with keys 'tile','j','i',
         indicating the ECCO grid cells contained in each cell. Each dictionary in the list corresponds to
         a cell in the given geometry (which may contain one or multiple ECCO grid cells).
       cells_all_single: bool, if True, process as if all cells are 1x1 (no nesting or grouping of grid cells).
       cell_impact_years_save: (optional) bool, if True, save cell impacts in individual years as well as variances.
         If False, function returns impact_forc_senststep_attimes (the spatially integrated cell_impact_attimes_atforc)
         instead of the full cell_impact_attimes_atforc (which can be a very large array).
       closest_forc_to_sens_atyears_multiyr: 3-D numpy datetime64 array, same as above but for
                                     multi-year sensitivity time steps
    """
    
    pass    
    
    import numpy as np
    import xarray as xr
    from os.path import join,expanduser
#     import time
#     time_log = [time.time()]
    # # # Define subfunctions used in forcsens_var_cells.py

    def preinit_forc_read(forc_source,forc_dir,curr_forc):
        if 'ecco' in forc_source:
            if 'closest_forc_to_sens_atyears_multiyr' in inputs.keys():
                curr_closest_forc_to_sens_atyears_array = np.concatenate((\
                        inputs['closest_forc_to_sens_atyears_multiyr'],\
                        inputs['closest_forc_to_sens_atyears']),axis=1)
            else:
                curr_closest_forc_to_sens_atyears_array = inputs['closest_forc_to_sens_atyears']
            if curr_forc == 'emp':
                forc_preinit_array = ecco_emp_forc_read_weekly(forc_dir,\
                                                                 years_to_predict,\
                                                                 inputs['max_weeks_lead'])
            else:
                forc_preinit_array = ecco_forc_read_weekly(curr_forc,\
                                                             forc_dir,\
                                                             years_to_predict,\
                                                             inputs['max_weeks_lead'],\
                                                         curr_closest_forc_to_sens_atyears_array)
        if 'era5' in forc_source:
            forc_preinit_array = era5_forc_read_weekly(curr_forc,\
                                                         forc_dir,\
                                                         years_to_predict,\
                                                         inputs['max_weeks_lead'])
        if 'jra55do' in forc_source:
            forc_preinit_array = jra55do_friver_forc_read_weekly(forc_dir,\
                                                                  years_to_predict,\
                                                                  inputs['max_weeks_lead'])

        return forc_preinit_array

    def postinit_forc_read(forc_source,forc_dir,curr_forc):
        if 'seas5_ensmean' in forc_source:
            # open SEAS5 ensemble means
            forc_postinit_array = \
                   seas5_ensmean_forc_read_weekly(curr_forc,\
                                             forc_dir,\
                                             years_to_predict,\
                                             inputs['max_weeks_lead'])
            n_postinit_leads = 8
            forc_postinit_time_spacing = 'weekly'
        if 'cesm_dple_ensmean' in forc_source:
            # open CESM DPLE ensemble means
            forc_postinit_array = \
                   cesm_dple_ensmean_forc_read_monthly(curr_forc,\
                                             forc_dir,\
                                             years_to_predict,\
                                             inputs['max_weeks_lead'])
            n_postinit_leads = 6
            forc_postinit_time_spacing = 'monthly'
        if 'cesm_hrdp_ensmean' in forc_source:
            # open CESM HRDP ensemble means
            forc_postinit_array = \
                   cesm_hrdp_ensmean_forc_read_monthly(curr_forc,\
                                             forc_dir,\
                                             years_to_predict,\
                                             inputs['max_weeks_lead'])
            n_postinit_leads = 3
            forc_postinit_time_spacing = 'monthly'

        return forc_postinit_array,n_postinit_leads,forc_postinit_time_spacing

    
    def expand_arrays(curr_array,n_tsteps_to_add,dimnum_to_expand,fill_value=None):
        pad_widths = []
        for dimnum in range(len(curr_array.shape)):
            if ((dimnum == dimnum_to_expand)\
              or (dimnum == len(curr_array.shape) + dimnum_to_expand)):
                pad_widths.append((n_tsteps_to_add,0))
            else:
                pad_widths.append((0,0))
        if fill_value is None:
            if curr_array.dtype == '<M8[ns]':
                fill_value = np.datetime64('NaT')
            else:
                fill_value = 0
        curr_array = np.pad(curr_array,\
                            pad_width=tuple(pad_widths),\
                            mode='constant',constant_values=fill_value)

        return curr_array
    

    def preinit_forcing_index_arrays(inputs,sens_shape0,closest_forc_to_sens_atyears_currmo,\
                                     forc_preinit_times_dict):
        # # assemble preinit forcing index arrays
        good_years_ind = np.empty((0,),dtype=np.int64)
        curr_t_ind = np.empty((0,),dtype=np.int64)
        if 'preinit_subt' in forc_preinit_times_dict.keys():
            good_years_ind_subt = np.empty((0,),dtype=np.int64)
            curr_t_ind_subt = np.empty((0,),dtype=np.int64)
        for year_count,forc_time in enumerate(\
                                    closest_forc_to_sens_atyears_currmo[tstep_count,:]):
            if tstep_count < sens_shape0 - 1:
                if np.abs(closest_forc_to_sens_atyears_currmo[tstep_count+1,year_count] \
                  - forc_time) < np.timedelta64(3,'D'):
                    # avoid duplication of same time point
                    continue
            
            curr_t = (np.abs(forc_preinit_times_dict['preinit'] - forc_time) < np.timedelta64(3,'D')).nonzero()[0]
            if len(curr_t) == 1:
                good_years_ind = np.hstack((good_years_ind,year_count))
                curr_t_ind = np.hstack((curr_t_ind,curr_t))
            
            if 'preinit_subt' in forc_preinit_times_dict.keys():
                curr_t = (np.abs(forc_preinit_times_dict['preinit_subt'] - forc_time) < np.timedelta64(3,'D'))\
                                .nonzero()[0]
                if len(curr_t) == 1:
                    good_years_ind_subt = np.hstack((good_years_ind_subt,year_count))
                    curr_t_ind_subt = np.hstack((curr_t_ind_subt,curr_t))

        outputs = [good_years_ind,curr_t_ind]
        if 'preinit_subt' in forc_preinit_times_dict.keys():
            outputs += [good_years_ind_subt,curr_t_ind_subt]

        return outputs


    def postinit_forcing_index_arrays(inputs,sens_shape0,closest_forc_to_sens_atyears_currmo,\
                                      forc_postinit_dict):
        # # assemble postinit forcing index arrays
        good_years_ind = np.empty((0,),dtype=np.int64)
        init_ind = np.empty((forc_postinit_dict['n_postinit_leads'],0),dtype=np.int64)        
        t_ind = np.empty((forc_postinit_dict['n_postinit_leads'],0),dtype=np.int64)
        postinit_lead_ind = np.empty((forc_postinit_dict['n_postinit_leads'],0),dtype=np.int64)
        for year_count,forc_time in enumerate(\
          closest_forc_to_sens_atyears_currmo[tstep_count,:]):
            if tstep_count < sens_shape0 - 1:
                if np.abs(closest_forc_to_sens_atyears_currmo[tstep_count+1,year_count] \
                  - forc_time) < np.timedelta64(3,'D'):
                    # avoid duplication of same time point
                    continue

            if forc_postinit_dict['time_spacing'] == 'monthly':
                curr_init_ind,curr_t_ind = (np.abs(\
                                  forc_postinit_dict['pred_time'].astype('datetime64[M]')\
                                   - forc_time.astype('datetime64[M]'))\
                                               < np.timedelta64(1,'M')).nonzero()                
            else:     
                curr_init_ind,curr_t_ind = (np.abs(\
                                  forc_postinit_dict['pred_time']\
                                   - forc_time) < np.timedelta64(3,'D')).nonzero()
            if len(curr_t_ind) == 0:
                continue

            # sort lead times associated with forcing predictions
            init_time_spacing = np.nanmean(np.diff(forc_postinit_dict['init_time']))
            curr_postinit_leads = np.empty((len(curr_init_ind),),dtype='timedelta64[ns]')
            curr_postinit_leads.fill('NaT')
            for lead_count,(curr_init,curr_t) in enumerate(zip(curr_init_ind,curr_t_ind)):                
                curr_postinit_leads[lead_count] = forc_postinit_dict['pred_time'][curr_init,curr_t]\
                                                         - forc_postinit_dict['init_time'][curr_init]
            min_postinit_leads = np.nanmin(curr_postinit_leads)
            curr_postinit_lead_ind = (np.round((curr_postinit_leads - min_postinit_leads)/init_time_spacing)\
                                        + np.floor(min_postinit_leads/init_time_spacing)).astype('int64')

            padding_len = forc_postinit_dict['n_postinit_leads']-len(curr_init_ind)
            good_years_ind = np.append(good_years_ind,np.array([year_count]),axis=-1)
            init_ind = np.append(init_ind,\
                                 np.pad(np.reshape(curr_init_ind,(-1,1)),\
                                        pad_width=((0,padding_len),(0,0)),\
                                        mode='constant',constant_values=-1000),axis=-1)
            t_ind = np.append(t_ind,\
                                 np.pad(np.reshape(curr_t_ind,(-1,1)),\
                                        pad_width=((0,padding_len),(0,0)),\
                                        mode='constant',constant_values=-1000),axis=-1)
            postinit_lead_ind = np.append(postinit_lead_ind,\
                                 np.pad(np.reshape(curr_postinit_lead_ind,(-1,1)),\
                                        pad_width=((0,padding_len),(0,0)),\
                                        mode='constant',constant_values=-1000),axis=-1)

        outputs = [good_years_ind,init_ind,t_ind,postinit_lead_ind]
        
        return outputs


    
    def forc_cumsum_attstep(curr_forc_attime,cumsum_mask_forc,cumsum_mask):
        mask = nanmask_create(curr_forc_attime)
        curr_forc_attime[mask < 0.01] = 0
        # cumulatively sum impact and mask
        cumsum_mask_forc += (mask*curr_forc_attime)
        cumsum_mask += mask

        return cumsum_mask_forc,cumsum_mask
    

    def forc_cumsum_in_t_range(chunk_array,chunk_start,chunk_end,\
                                   t_ind_attstep,\
                                   cumsum_mask_forc_attstep,cumsum_mask_attstep):
        curr_in_t_range = np.logical_and(t_ind_attstep >= chunk_start,\
                                         t_ind_attstep < chunk_end)\
                                         .nonzero()[0]
        for curr_in_t in curr_in_t_range:
            curr_t = t_ind_attstep[curr_in_t]
            cumsum_mask_forc_attstep,cumsum_mask_attstep = \
                    forc_cumsum_attstep(chunk_array[curr_t - chunk_start,:,:,:],\
                                        cumsum_mask_forc_attstep,\
                                        cumsum_mask_attstep)

        return cumsum_mask_forc_attstep,cumsum_mask_attstep
        

    def forc_cumsum_in_t_range_postinit(chunk_array,chunk_start,chunk_end,\
                                           init_ind_attstep,\
                                           t_ind_attstep,\
                                           lead_ind_attstep,\
                                           cumsum_mask_forc_attstep,cumsum_mask_attstep):
        curr_in_init_range = np.logical_and(init_ind_attstep >= chunk_start,\
                                             init_ind_attstep < chunk_end)\
                                             .nonzero()
        for lead_curr,yr_curr in zip(curr_in_init_range[0],curr_in_init_range[1]):
            curr_init = init_ind_attstep[lead_curr,yr_curr]
            curr_t = t_ind_attstep[lead_curr,yr_curr]
            curr_lead = lead_ind_attstep[lead_curr,yr_curr]
            curr_cumsum_mask_forc_attstep,curr_cumsum_mask_attstep = \
                    forc_cumsum_attstep(chunk_array[curr_init - chunk_start,curr_t,:,:,:],\
                                        cumsum_mask_forc_attstep[curr_lead,:,:,:],\
                                        cumsum_mask_attstep[curr_lead,:,:,:])
            cumsum_mask_forc_attstep[curr_lead,:,:,:] = curr_cumsum_mask_forc_attstep
            cumsum_mask_attstep[curr_lead,:,:,:] = curr_cumsum_mask_attstep

        return cumsum_mask_forc_attstep,cumsum_mask_attstep
    


    def sensadj_apply(curr_forc_num,sensadj_ratios,curr_impact_attime):
        curr_sensadj_ratios = sensadj_ratios[curr_forc_num,pred_count,tstep_count]
        curr_impact_attime = curr_sensadj_ratios*curr_impact_attime

        return curr_impact_attime
    

    @jit(nopython=True)
    def cell_aggregate_jit(curr_impact_attstep,tile_ind,j_ind,i_ind):
        curr_cell_impact_attimes = np.zeros((curr_impact_attstep.shape[0],),dtype=np.float64)
        for tile,j,i in zip(tile_ind,j_ind,i_ind):
            curr_cell_impact = curr_impact_attstep[:,tile,j,i]
            curr_cell_impact_attimes += curr_cell_impact

        return curr_cell_impact_attimes

    @jit(nopython=True)
    def cell_impacts_reshape_allsingle(curr_impact_attstep,tile_ind,j_ind,i_ind):
        cell_impact_attstep = np.empty((curr_impact_attstep.shape[0],len(tile_ind)),dtype=np.float64)
        cell_impact_attstep.fill(np.nan)
        for cell_num,(tile,j,i) in enumerate(zip(tile_ind,j_ind,i_ind)):
            cell_impact_attstep[:,cell_num] = curr_impact_attstep[:,tile,j,i]

        return cell_impact_attstep    
    
    def cell_impacts_aggregate(curr_impact_attstep,cell_indices):
        curr_impact_attstep[np.isnan(curr_impact_attstep)] = 0
        cell_impact_attstep = np.empty((curr_impact_attstep.shape[0],len(cell_indices)),dtype=np.float64)
        cell_impact_attstep.fill(np.nan)
        for cell_num,cell_ind_dict in enumerate(cell_indices):
            # aggregate impacts
            
            # curr_cell_impact_attimes = np.sum(curr_impact_attstep[(slice(None),)+\
            #                                   (cell_ind_dict['tile'],cell_ind_dict['j'],cell_ind_dict['i'])],\
            #                                   axis=-1)
            curr_cell_impact_attimes = cell_aggregate_jit(curr_impact_attstep,\
                                                          cell_ind_dict['tile'],\
                                                          cell_ind_dict['j'],\
                                                          cell_ind_dict['i'])
            cell_impact_attstep[:,cell_num] = curr_cell_impact_attimes
        
        cell_impact_attstep[np.abs(cell_impact_attstep) < 1.e-20] = np.nan

        return cell_impact_attstep    


    def cell_impacts_compute_aggregate(chunk_array,tmean_attstep,\
                                          chunk_start,chunk_end,\
                                          cell_impact_attstep,\
                                          sens_tstep,sensadj_opt_ratios,\
                                          good_years_ind_attstep,t_ind_attstep,\
                                          cells_all_single,cell_indices,cell_indices_np_list,tstep_count):
        curr_in_t_range = np.logical_and(t_ind_attstep >= chunk_start,\
                                         t_ind_attstep < chunk_end)\
                                         .nonzero()[0]
        curr_impact_attstep = np.zeros((len(curr_in_t_range),)+chunk_array.shape[-3:],dtype=np.float64)
        for in_t_count,curr_in_t in enumerate(curr_in_t_range):
            curr_t = t_ind_attstep[curr_in_t]
            curr_forc = chunk_array[curr_t - chunk_start,:,:,:] - tmean_attstep
            curr_impact_attime = sens_tstep*curr_forc
            if len(sensadj_opt_ratios) > 0:
                if sensadj_opt_ratios[0]:
                    curr_impact_attime = sensadj_opt_ratios[1]*curr_impact_attime
            curr_impact_attstep[in_t_count,:,:,:] = curr_impact_attime

        if cells_all_single:
            curr_t_cell_impact_attstep = cell_impacts_reshape_allsingle(curr_impact_attstep,\
                                                                        *cell_indices_np_list)
        else:
            curr_t_cell_impact_attstep = cell_impacts_aggregate(curr_impact_attstep,cell_indices)
        
        for in_t_count,curr_in_t in enumerate(curr_in_t_range):
            curr_good_year_ind = good_years_ind_attstep[curr_in_t]
            cell_impact_attstep[curr_good_year_ind,:] = curr_t_cell_impact_attstep[in_t_count,:]

        return cell_impact_attstep
        
    
    def cell_impacts_compute_aggregate_postinit(chunk_array,tmean_attstep,\
                                                  chunk_start,chunk_end,\
                                                  cell_impact_attstep,\
                                                  sens_tstep,sensadj_opt_ratios,\
                                                  good_years_ind_attstep,init_ind_attstep,\
                                                  t_ind_attstep,lead_ind_attstep,\
                                                  cells_all_single,cell_indices,cell_indices_np_list,tstep_count):
        curr_in_init_range = np.logical_and(init_ind_attstep >= chunk_start,\
                                             init_ind_attstep < chunk_end)\
                                             .nonzero()
        curr_impact_attstep = np.zeros((len(curr_in_init_range[0]),)+chunk_array.shape[-3:],dtype=np.float64)
        for in_t_count,(lead_curr,yr_curr) in enumerate(zip(curr_in_init_range[0],\
                                                             curr_in_init_range[1])):
            curr_init = init_ind_attstep[lead_curr,yr_curr]
            curr_t = t_ind_attstep[lead_curr,yr_curr]
            curr_lead = lead_ind_attstep[lead_curr,yr_curr]
            curr_forc = chunk_array[curr_init - chunk_start,curr_t,:,:,:] - tmean_attstep[curr_lead,:,:,:]
            curr_impact_attime = sens_tstep*curr_forc
            if len(sensadj_opt_ratios) > 0:
                if sensadj_opt_ratios[0]:
                    curr_impact_attime = sensadj_opt_ratios[1]*curr_impact_attime
            curr_impact_attstep[in_t_count,:,:,:] = curr_impact_attime

        if cells_all_single:
            curr_t_cell_impact_attstep = cell_impacts_reshape_allsingle(curr_impact_attstep,\
                                                                        *cell_indices_np_list)
        else:
            curr_t_cell_impact_attstep = cell_impacts_aggregate(curr_impact_attstep,cell_indices)
        for in_t_count,(lead_curr,yr_curr) in enumerate(zip(curr_in_init_range[0],\
                                                             curr_in_init_range[1])):
            curr_lead_ind = lead_ind_attstep[lead_curr,yr_curr]
            curr_good_year_ind = good_years_ind_attstep[lead_curr,yr_curr]
            cell_impact_attstep[curr_lead_ind,curr_good_year_ind,:] = curr_t_cell_impact_attstep[in_t_count,:]

        return cell_impact_attstep



    def cell_impacts_aggregate_allforc(curr_arrays_dict,curr_tmean_attstep_dict,\
                                         chunk_bounds_dict,\
                                         curr_cell_impact_attstep_dict,\
                                         sens_tstep,sensadj_opt_ratios_dict,\
                                         good_years_ind_attstep_dict,t_ind_attstep_dict,\
                                         cells_all_single,cell_indices,cell_indices_np_list,tstep_count):

        preinit_chunk_array = curr_arrays_dict['preinit']
        preinit_tmean_attstep = curr_tmean_attstep_dict['preinit']
        pre_chunk_start,pre_chunk_end = chunk_bounds_dict['preinit']
        preinit_cell_impact_attstep = curr_cell_impact_attstep_dict['preinit']
        preinit_good_years_ind_attstep = good_years_ind_attstep_dict['preinit']
        preinit_t_ind_attstep = t_ind_attstep_dict['preinit']
        
        preinit_cell_impact_attstep = cell_impacts_compute_aggregate(preinit_chunk_array,preinit_tmean_attstep,\
                                                                      pre_chunk_start,pre_chunk_end,\
                                                                      preinit_cell_impact_attstep,\
                                                                      sens_tstep,sensadj_opt_ratios_dict['preinit'],\
                                                                      preinit_good_years_ind_attstep,\
                                                                      preinit_t_ind_attstep,\
                                                                      cells_all_single,cell_indices,\
                                                                      cell_indices_np_list,tstep_count)

        if 'preinit_runoff' in curr_arrays_dict.keys():
            preinit_r_chunk_array = curr_arrays_dict['preinit_runoff']
            preinit_r_tmean_attstep = curr_tmean_attstep_dict['preinit_runoff']
            pre_r_chunk_start,pre_r_chunk_end = chunk_bounds_dict['preinit_runoff']
            preinit_r_cell_impact_attstep = curr_cell_impact_attstep_dict['preinit_runoff'] + 0
            preinit_r_good_years_ind_attstep = good_years_ind_attstep_dict['preinit_runoff']
            preinit_r_t_ind_attstep = t_ind_attstep_dict['preinit_runoff']
            
            preinit_r_cell_impact_attstep = cell_impacts_compute_aggregate(preinit_r_chunk_array,preinit_r_tmean_attstep,\
                                                                          pre_r_chunk_start,pre_r_chunk_end,\
                                                                          preinit_r_cell_impact_attstep,\
                                                                          sens_tstep,sensadj_opt_ratios_dict['preinit'],\
                                                                          preinit_r_good_years_ind_attstep,\
                                                                          preinit_r_t_ind_attstep,\
                                                                          cells_all_single,cell_indices,\
                                                                          cell_indices_np_list,tstep_count)
            
            preinit_cell_impact_attstep += \
                np.where(np.logical_and(np.isnan(curr_cell_impact_attstep_dict['preinit_runoff']),\
                                        ~np.isnan(preinit_r_cell_impact_attstep)),\
                         preinit_r_cell_impact_attstep,0)
            
        cell_impact_arrays = {'preinit':preinit_cell_impact_attstep}
        
        if 'preinit_subt' in curr_arrays_dict.keys():
            preinit_subt_chunk_array = curr_arrays_dict['preinit_subt']
            preinit_subt_tmean_attstep = curr_tmean_attstep_dict['preinit_subt']
            presub_chunk_start,presub_chunk_end = chunk_bounds_dict['preinit_subt']
            preinit_subt_cell_impact_attstep = curr_cell_impact_attstep_dict['preinit_subt']
            preinit_subt_good_years_ind_attstep = good_years_ind_attstep_dict['preinit_subt']
            preinit_subt_t_ind_attstep = t_ind_attstep_dict['preinit_subt']

            preinit_subt_cell_impact_attstep = cell_impacts_compute_aggregate(preinit_subt_chunk_array,\
                                                                              preinit_subt_tmean_attstep,\
                                                                              presub_chunk_start,presub_chunk_end,\
                                                                              preinit_subt_cell_impact_attstep,\
                                                                              sens_tstep,\
                                                                              sensadj_opt_ratios_dict['preinit_subt'],\
                                                                              preinit_subt_good_years_ind_attstep,\
                                                                              preinit_subt_t_ind_attstep,\
                                                                              cells_all_single,cell_indices,\
                                                                              cell_indices_np_list,tstep_count)
            cell_impact_arrays['preinit_subt'] = preinit_subt_cell_impact_attstep
            
        if 'postinit' in curr_arrays_dict.keys():
            postinit_chunk_array = curr_arrays_dict['postinit']
            postinit_tmean_attstep = curr_tmean_attstep_dict['postinit']
            post_chunk_start,post_chunk_end = chunk_bounds_dict['postinit']
            postinit_cell_impact_attstep = curr_cell_impact_attstep_dict['postinit']
            postinit_good_years_ind_attstep = good_years_ind_attstep_dict['postinit']
            postinit_init_ind_attstep = t_ind_attstep_dict['postinit_init']
            postinit_t_ind_attstep = t_ind_attstep_dict['postinit_t']
            postinit_lead_ind_attstep = t_ind_attstep_dict['postinit_lead']

            postinit_cell_impact_attstep = cell_impacts_compute_aggregate_postinit(postinit_chunk_array,\
                                                                      postinit_tmean_attstep,\
                                                                      post_chunk_start,post_chunk_end,\
                                                                      postinit_cell_impact_attstep,\
                                                                      sens_tstep,sensadj_opt_ratios_dict['postinit'],\
                                                                      postinit_good_years_ind_attstep,\
                                                                      postinit_init_ind_attstep,\
                                                                      postinit_t_ind_attstep,\
                                                                      postinit_lead_ind_attstep,\
                                                                      cells_all_single,cell_indices,\
                                                                      cell_indices_np_list,tstep_count)
            cell_impact_arrays['postinit'] = postinit_cell_impact_attstep
                        
        if 'postinit_subt' in curr_arrays_dict.keys():
            postinit_subt_chunk_array = curr_arrays_dict['postinit_subt']
            postinit_subt_tmean_attstep = curr_tmean_attstep_dict['postinit_subt']
            postsub_chunk_start,postsub_chunk_end = chunk_bounds_dict['postinit_subt']
            postinit_subt_cell_impact_attstep = curr_cell_impact_attstep_dict['postinit_subt']
            postinit_subt_good_years_ind_attstep = good_years_ind_attstep_dict['postinit_subt']
            postinit_subt_init_ind_attstep = t_ind_attstep_dict['postinit_subt_init']
            postinit_subt_t_ind_attstep = t_ind_attstep_dict['postinit_subt_t']
            postinit_subt_lead_ind_attstep = t_ind_attstep_dict['postinit_subt_lead']

            postinit_subt_cell_impact_attstep = cell_impacts_compute_aggregate_postinit(postinit_subt_chunk_array,\
                                                                          postinit_subt_tmean_attstep,\
                                                                          postsub_chunk_start,postsub_chunk_end,\
                                                                          postinit_subt_cell_impact_attstep,\
                                                                          sens_tstep,\
                                                                          sensadj_opt_ratios_dict['postinit_subt'],\
                                                                          postinit_subt_good_years_ind_attstep,\
                                                                          postinit_subt_init_ind_attstep,\
                                                                          postinit_subt_t_ind_attstep,\
                                                                          postinit_subt_lead_ind_attstep,\
                                                                          cells_all_single,cell_indices,\
                                                                          cell_indices_np_list,tstep_count)
            cell_impact_arrays['postinit_subt'] = postinit_subt_cell_impact_attstep
        
        return cell_impact_arrays

    
    # # # End of subfunction definitions
    
    
        
    curr_forc = inputs['curr_forc']
    adj_sens_loc = inputs['adj_sens_loc']
    years_to_predict = inputs['years_to_predict']
    cell_indices = inputs['cell_indices']
    n_cells = len(cell_indices)
    
    trend_remove = False
    if 'trend_remove' in inputs.keys():
        if inputs['trend_remove'] == True:
            trend_remove = True

    if ('obs' in inputs.keys()) or ('reconstr' in inputs.keys()) or ('model' in inputs.keys()):
        resid_compute = True
    else:
        resid_compute = False
    
    if inputs['cells_all_single']:
        # create cell indexing tuple
        tile_ind = np.empty((n_cells,),dtype=np.int32)
        j_ind = np.empty((n_cells,),dtype=np.int32)
        i_ind = np.empty((n_cells,),dtype=np.int32)
        for cell_num,cell_ind_dict in enumerate(cell_indices):
            tile_ind[cell_num] = cell_ind_dict['tile'][0]
            j_ind[cell_num] = cell_ind_dict['j'][0]
            i_ind[cell_num] = cell_ind_dict['i'][0]
        cell_indices_np_list = [tile_ind,j_ind,i_ind]
    else:
        cell_indices_np_list = []
    
    
    # # read/load preinit forcing arrays
    
    add_runoff = False
    if inputs['curr_forc'] == 'empmr':
        forc_preinit_array = preinit_forc_read(inputs['forcing_preinit']['forc_type'],\
                                               inputs['forcing_preinit']['forc_ep_dir'],\
                                               'emp')
        if 'forc_runoff_type' in inputs['forcing_preinit'].keys():
            add_runoff = True
            forc_preinit_r_array = preinit_forc_read(inputs['forcing_preinit']['forc_runoff_type'],\
                                                     inputs['forcing_preinit']['forc_runoff_dir'],\
                                                     inputs['curr_forc'])
    else:
        forc_preinit_array = preinit_forc_read(inputs['forcing_preinit']['forc_type'],\
                                               inputs['forcing_preinit']['forc_dir'],\
                                               inputs['curr_forc'])
    
    
    if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
        forc_preinit_subtract_array = preinit_forc_read(inputs['forcing_preinit']['forc_subtract_type'],\
                                                        inputs['forcing_preinit']['forc_subtract_dir'],\
                                                        inputs['curr_forc'])
        
    
    # # read/load postinit forcing (prediction model) arrays
    
    if 'forcing_postinit' in inputs.keys():
        forc_postinit_array,n_postinit_leads,forc_postinit_time_spacing = \
                    postinit_forc_read(inputs['forcing_postinit']['forc_type'],\
                                       inputs['forcing_postinit']['forc_dir'],\
                                       inputs['curr_forc'])
        if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
            forc_postinit_subtract_array,n_postinit_subt_leads,forc_postinit_subt_time_spacing = \
                        postinit_forc_read(inputs['forcing_postinit']['forc_subtract_type'],\
                                           inputs['forcing_postinit']['forc_subtract_dir'],\
                                           inputs['curr_forc'])

    # # allocate arrays for all prediction months
    
    arrays_to_expand = {'-1':{},'-2':{},'-3':{}}
    sens_times = np.empty(inputs['sens_time'].shape,dtype='datetime64[ns]')
    arrays_to_expand['-1']['sens_times'] = sens_times
    closest_forc_to_sens_atyears = np.empty(inputs['closest_forc_to_sens_atyears'].shape,\
                                            dtype='datetime64[ns]')
    arrays_to_expand['-2']['closest_forc_to_sens_atyears'] = closest_forc_to_sens_atyears
    
    len_tsteps_dim = closest_forc_to_sens_atyears.shape[-2]
    preinit_impact_senststep_attimes = np.zeros((12,len_tsteps_dim,len(years_to_predict)))
    arrays_to_expand['-2']['preinit_impact_senststep_attimes'] = preinit_impact_senststep_attimes
    if 'forcing_postinit' in inputs.keys():
        postinit_impact_senststep_attimes = np.zeros((n_postinit_leads,12,len_tsteps_dim,len(years_to_predict)))
        arrays_to_expand['-2']['postinit_impact_senststep_attimes'] = postinit_impact_senststep_attimes
        postinit_init_times_array = np.empty((n_postinit_leads,12,len_tsteps_dim,len(years_to_predict)),\
                                             dtype='datetime64[ns]')
        postinit_init_times_array.fill(np.datetime64('NaT'))
        arrays_to_expand['-2']['postinit_init_times_array'] = postinit_init_times_array
    if inputs['cell_impact_years_save']:
        preinit_cell_impact_attimes_atforc = np.zeros((12,len_tsteps_dim,len(years_to_predict),n_cells))
        arrays_to_expand['-3']['preinit_cell_impact_attimes_atforc'] = preinit_cell_impact_attimes_atforc
        if 'forcing_postinit' in inputs.keys():
            postinit_cell_impact_attimes_atforc = \
                        np.zeros((n_postinit_leads,12,len_tsteps_dim,len(years_to_predict),n_cells))
            arrays_to_expand['-3']['postinit_cell_impact_attimes_atforc'] = postinit_cell_impact_attimes_atforc
        if resid_compute:
            if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                preinit_subt_cell_impact_attimes_atforc = np.zeros((12,len_tsteps_dim,len(years_to_predict),n_cells))
                arrays_to_expand['-3']['preinit_subt_cell_impact_attimes_atforc'] = \
                                        preinit_subt_cell_impact_attimes_atforc
            if 'forcing_postinit' in inputs.keys():
                if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                    postinit_subt_cell_impact_attimes_atforc = \
                      np.zeros((n_postinit_subt_leads,12,len_tsteps_dim,len(years_to_predict),n_cells))
                    arrays_to_expand['-3']['postinit_subt_cell_impact_attimes_atforc'] = \
                                            postinit_subt_cell_impact_attimes_atforc
        else:
            cell_impact_var_atforc = np.zeros((12,len_tsteps_dim,n_cells))
            arrays_to_expand['-2']['cell_impact_var_atforc'] = cell_impact_var_atforc
            cum_cell_impact_var_atforc = np.zeros((12,len_tsteps_dim,n_cells))
            arrays_to_expand['-2']['cum_cell_impact_var_atforc'] = cum_cell_impact_var_atforc
    
    if resid_compute:
        if 'forcing_postinit' in inputs.keys():
            resid_cell_impact_var_atforc = np.empty((n_postinit_leads,12,len_tsteps_dim,n_cells))
            cum_resid_cell_impact_var_atforc = np.empty((n_postinit_leads,12,len_tsteps_dim,n_cells))
        else:
            resid_cell_impact_var_atforc = np.empty((12,len_tsteps_dim,n_cells))
            cum_resid_cell_impact_var_atforc = np.empty((12,len_tsteps_dim,n_cells))
        resid_cell_impact_var_atforc.fill(np.nan)
        arrays_to_expand['-2']['resid_cell_impact_var_atforc'] = resid_cell_impact_var_atforc        
        cum_resid_cell_impact_var_atforc.fill(np.nan)
        arrays_to_expand['-2']['cum_resid_cell_impact_var_atforc'] = cum_resid_cell_impact_var_atforc

    
    
    # # if sensitivity adjustments are involved, load sensadj ratios and prepare resid var array for each pred month, tstep
    
    sensadj_compute = False
    if 'sensadj' in inputs['forcing_preinit']['forc_type']:
        sensadj_compute = True
    if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
        if 'sensadj' in inputs['forcing_preinit']['forc_subtract_type']:
            sensadj_compute = True
    if 'forcing_postinit' in inputs.keys():
        if 'sensadj' in inputs['forcing_postinit']['forc_type']:
            sensadj_compute = True
        if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
            if 'sensadj' in inputs['forcing_postinit']['forc_subtract_type']:
                sensadj_compute = True
    if sensadj_compute == True:
        curr_file = inputs['sensadj_params']['sensadj_ratios_file']
        ds_sensadj = xr.open_mfdataset(curr_file,engine='zarr')
        sensadj_forc_nums = ds_sensadj.forc.values
        sensadj_ratios = ds_sensadj.sens_adj_ratios.transpose('forc','pred_month','sens_tval').values
        resid_var_inpart = np.empty((12,55))
        resid_var_inpart.fill(np.nan)        
    
    if inputs['max_weeks_lead'] > 52:
        # load multi-year ECCO adjoint sensitivities
        sens_multiyr,_ = \
                adj_sens_read_multiyr(curr_forc,adj_sens_loc)
        sens_multiyr = sens_multiyr[\
          -inputs['closest_forc_to_sens_atyears_multiyr'].shape[1]:,:,:,:]

    
    for pred_count,pred_month in enumerate(range(1,13)):
        
        # load ECCO adjoint sensitivities for current prediction month       
        sens,_ = adj_sens_read(curr_forc,adj_sens_loc,pred_month)

        sens_times_currmo = inputs['sens_time'][pred_count,:]
        closest_forc_to_sens_atyears_currmo = \
          inputs['closest_forc_to_sens_atyears'][pred_count,:,:]

        sens_end_offset = sens_times_currmo.shape[0] - sens.shape[0]
        
        if inputs['max_weeks_lead'] > 52:
            # concatenate multi-year to seasonal sensitivities
            
            n_before_sens_seas = np.sum(\
              inputs['closest_forc_to_sens_atyears_multiyr'][pred_count,:,:]\
              - inputs['closest_forc_to_sens_atyears'][pred_count,[0],:]\
              < -np.timedelta64(3,'D'),axis=0)
            multiyr_prepend_len = np.int32(np.round(\
              np.nanmean(n_before_sens_seas.astype('float64'))))
            sens_multiyr_prepend = sens_multiyr[:multiyr_prepend_len,:,:,:]
            sens_times_prepend = inputs['sens_time_multiyr'][:multiyr_prepend_len]
            closest_forc_to_sens_atyears_prepend = inputs\
              ['closest_forc_to_sens_atyears_multiyr'][pred_count,:multiyr_prepend_len,:]            

            # expand arrays as needed
            n_tsteps_to_add = multiyr_prepend_len \
              + closest_forc_to_sens_atyears_currmo.shape[0] \
              - preinit_impact_senststep_attimes.shape[1]
            if n_tsteps_to_add > 0:
                dimnum_to_expand = -1
                sens_times = expand_arrays(sens_times,n_tsteps_to_add,dimnum_to_expand)
                
                dimnum_to_expand = -2                
                closest_forc_to_sens_atyears = expand_arrays(closest_forc_to_sens_atyears,\
                                                             n_tsteps_to_add,dimnum_to_expand)
                preinit_impact_senststep_attimes = expand_arrays(preinit_impact_senststep_attimes,\
                                                                 n_tsteps_to_add,dimnum_to_expand)
                if 'forcing_postinit' in inputs.keys():
                    postinit_impact_senststep_attimes = expand_arrays(postinit_impact_senststep_attimes,\
                                                                      n_tsteps_to_add,dimnum_to_expand)
                    postinit_init_times_array = expand_arrays(postinit_init_times_array,\
                                                              n_tsteps_to_add,dimnum_to_expand)
                if resid_compute:
                    resid_cell_impact_var_atforc = expand_arrays(resid_cell_impact_var_atforc,\
                                                                 n_tsteps_to_add,dimnum_to_expand,\
                                                                 fill_value=np.nan)
                    cum_resid_cell_impact_var_atforc = expand_arrays(cum_resid_cell_impact_var_atforc,\
                                                                     n_tsteps_to_add,dimnum_to_expand,\
                                                                     fill_value=np.nan)
                else:
                    if inputs['cell_impact_years_save']:
                        cell_impact_var_atforc = expand_arrays(cell_impact_var_atforc,\
                                                               n_tsteps_to_add,dimnum_to_expand)
                        cum_cell_impact_var_atforc = expand_arrays(cum_cell_impact_var_atforc,\
                                                                   n_tsteps_to_add,dimnum_to_expand)                    
                
                dimnum_to_expand = -3
                if inputs['cell_impact_years_save']:
                    preinit_cell_impact_attimes_atforc = expand_arrays(preinit_cell_impact_attimes_atforc,\
                                                                       n_tsteps_to_add,dimnum_to_expand)
                    if 'forcing_postinit' in inputs.keys():
                        postinit_cell_impact_attimes_atforc = expand_arrays(postinit_cell_impact_attimes_atforc,\
                                                                            n_tsteps_to_add,dimnum_to_expand)
                    if resid_compute:
                        if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                            preinit_subt_cell_impact_attimes_atforc = expand_arrays(\
                                preinit_subt_cell_impact_attimes_atforc,n_tsteps_to_add,dimnum_to_expand)
                        if 'forcing_postinit' in inputs.keys():
                            if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                                postinit_subt_cell_impact_attimes_atforc = expand_arrays(\
                                    postinit_subt_cell_impact_attimes_atforc,n_tsteps_to_add,dimnum_to_expand)              


            # apply taper to seasonal sensitivities
            taper = np.expand_dims(np.arange(0.5,inputs['multiyr_seas_taper_weeks'],1)/\
              inputs['multiyr_seas_taper_weeks'],axis=(1,2,3))
            sens[:taper.shape[0],:,:,:] = \
              ((1 - taper)*sens_multiyr[multiyr_prepend_len:\
              (multiyr_prepend_len+taper.shape[0]),:,:,:])\
              + (taper*sens[:taper.shape[0],:,:,:])

            sens = np.concatenate((sens_multiyr_prepend,sens),axis=0)
            sens_times_currmo = np.concatenate(\
              (sens_times_prepend,sens_times_currmo),\
               axis=0)
            closest_forc_to_sens_atyears_currmo = np.concatenate(\
              (closest_forc_to_sens_atyears_prepend,\
              closest_forc_to_sens_atyears_currmo),axis=0)
        
        
        
        # allocate arrays for current prediction month
        len_tsteps_dim = closest_forc_to_sens_atyears_currmo.shape[0]
        preinit_cell_impact_attimes = np.empty((len_tsteps_dim,len(years_to_predict),n_cells))
        preinit_cell_impact_attimes.fill(np.nan)
        if add_runoff:
            preinit_r_cell_impact_attimes = np.empty((len_tsteps_dim,len(years_to_predict),n_cells))
            preinit_r_cell_impact_attimes.fill(np.nan)
        cum_preinit_cell_impact_attimes = np.empty((len_tsteps_dim,len(years_to_predict),n_cells))
        cum_preinit_cell_impact_attimes.fill(np.nan)
        if 'forcing_postinit' in inputs.keys():
            postinit_cell_impact_attimes = np.empty((n_postinit_leads,len_tsteps_dim,len(years_to_predict),n_cells))
            postinit_cell_impact_attimes.fill(np.nan)
            cum_postinit_cell_impact_attimes = np.zeros((n_postinit_leads,len_tsteps_dim,len(years_to_predict),n_cells))
        if ('obs' in inputs.keys()) or ('reconstr' in inputs.keys()):
            if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                preinit_subt_cell_impact_attimes = np.empty((len_tsteps_dim,len(years_to_predict),n_cells))
                preinit_subt_cell_impact_attimes.fill(np.nan)
                cum_preinit_subt_cell_impact_attimes = np.empty((len_tsteps_dim,len(years_to_predict),n_cells))
                cum_preinit_subt_cell_impact_attimes.fill(np.nan)
            if 'forcing_postinit' in inputs.keys():
                if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                    postinit_subt_cell_impact_attimes = \
                      np.empty((n_postinit_subt_leads,len_tsteps_dim,len(years_to_predict),n_cells))
                    postinit_subt_cell_impact_attimes.fill(np.nan)
                    cum_postinit_subt_cell_impact_attimes = \
                      np.zeros((n_postinit_subt_leads,len_tsteps_dim,len(years_to_predict),n_cells))

        # loading preinit arrays into memory
        forc_preinit_times = forc_preinit_array.time.values
        forc_preinit_np_array = forc_preinit_array.values
        if add_runoff:
            forc_preinit_r_times = forc_preinit_r_array.time.values
            forc_preinit_r_np_array = forc_preinit_r_array.values            
        if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
            forc_preinit_subtract_times = forc_preinit_subtract_array.time.values
            forc_preinit_subtract_np_array = forc_preinit_subtract_array.values

        
        # # create forcing index arrays
        
        preinit_good_years_ind_alltstep = np.empty(closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
        preinit_good_years_ind_alltstep.fill(-1000)
        preinit_t_ind_alltstep = np.empty(closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
        preinit_t_ind_alltstep.fill(-1000)
        forc_preinit_times_dict = {'preinit':forc_preinit_times}
        if add_runoff:
            preinit_r_good_years_ind_alltstep = np.empty(closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
            preinit_r_good_years_ind_alltstep.fill(-1000)
            preinit_r_t_ind_alltstep = np.empty(closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
            preinit_r_t_ind_alltstep.fill(-1000)
            forc_preinit_r_times_dict = {'preinit':forc_preinit_r_times}            
        if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
            preinit_subt_good_years_ind_alltstep = np.empty(closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
            preinit_subt_good_years_ind_alltstep.fill(-1000)
            preinit_subt_t_ind_alltstep = np.empty(closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
            preinit_subt_t_ind_alltstep.fill(-1000)
            forc_preinit_times_dict['preinit_subtract'] = forc_preinit_subtract_times
        if 'forcing_postinit' in inputs.keys():
            postinit_good_years_ind_alltstep = np.empty((n_postinit_leads,)\
                                                        +closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
            postinit_good_years_ind_alltstep.fill(-1000)
            postinit_init_ind_alltstep = np.empty((n_postinit_leads,)\
                                                  +closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
            postinit_init_ind_alltstep.fill(-1000)
            postinit_t_ind_alltstep = np.empty((n_postinit_leads,)\
                                               +closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
            postinit_t_ind_alltstep.fill(-1000)
            postinit_lead_ind_alltstep = np.empty((n_postinit_leads,)\
                                                  +closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
            postinit_lead_ind_alltstep.fill(-1000)
            forc_postinit_init_time = forc_postinit_array.init_time.values
            forc_postinit_pred_time = forc_postinit_array.pred_time.values
            forc_postinit_dict = {'postinit':{'init_time':forc_postinit_init_time,\
                                              'pred_time':forc_postinit_pred_time,\
                                              'n_postinit_leads':n_postinit_leads,\
                                              'time_spacing':forc_postinit_time_spacing}}
            if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                postinit_subt_good_years_ind_alltstep = np.empty((n_postinit_leads,)\
                                                                 +closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
                postinit_subt_good_years_ind_alltstep.fill(-1000)
                postinit_subt_init_ind_alltstep = np.empty((n_postinit_leads,)\
                                                        +closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
                postinit_subt_init_ind_alltstep.fill(-1000)
                postinit_subt_t_ind_alltstep = np.empty((n_postinit_leads,)\
                                                        +closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
                postinit_subt_t_ind_alltstep.fill(-1000)
                postinit_subt_lead_ind_alltstep = np.empty((n_postinit_leads,)\
                                                           +closest_forc_to_sens_atyears_currmo.shape,dtype=np.int64)
                postinit_subt_lead_ind_alltstep.fill(-1000)
                forc_postinit_subt_init_time = forc_postinit_subtract_array.init_time.values
                forc_postinit_subt_pred_time = forc_postinit_subtract_array.pred_time.values
                forc_postinit_dict['postinit_subt'] = {'init_time':forc_postinit_subt_init_time,\
                                                       'pred_time':forc_postinit_subt_pred_time,\
                                                       'n_postinit_leads':n_postinit_subt_leads,\
                                                       'time_spacing':forc_postinit_subt_time_spacing}
        sens_shape0 = sens.shape[0]
        for tstep_count in range(sens_shape0):
            curr_outputs = preinit_forcing_index_arrays(inputs,sens_shape0,closest_forc_to_sens_atyears_currmo,\
                                                        forc_preinit_times_dict)

            preinit_good_years_ind_alltstep[tstep_count,:len(curr_outputs[0])] = curr_outputs[0]
            preinit_t_ind_alltstep[tstep_count,:len(curr_outputs[1])] = curr_outputs[1]
            if add_runoff:
                curr_outputs_r = preinit_forcing_index_arrays(inputs,sens_shape0,closest_forc_to_sens_atyears_currmo,\
                                                        forc_preinit_r_times_dict)

                preinit_r_good_years_ind_alltstep[tstep_count,:len(curr_outputs_r[0])] = curr_outputs_r[0]
                preinit_r_t_ind_alltstep[tstep_count,:len(curr_outputs_r[1])] = curr_outputs_r[1]
            if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                preinit_subt_good_years_ind_alltstep[tstep_count,:len(curr_outputs[2])] = curr_outputs[2]
                preinit_subt_t_ind_alltstep[tstep_count,:len(curr_outputs[3])] = curr_outputs[3]

            if 'forcing_postinit' in inputs.keys():
                curr_outputs = postinit_forcing_index_arrays(inputs,sens_shape0,closest_forc_to_sens_atyears_currmo,\
                                                             forc_postinit_dict['postinit'])
                for year_count in range(len(curr_outputs[0])):
                    postinit_good_years_ind_alltstep[:,tstep_count,year_count] = \
                                                            curr_outputs[0][year_count]
                    for n_ind,lead_ind in enumerate(curr_outputs[3][:,year_count]):
                        if lead_ind >= 0:
                            postinit_init_ind_alltstep[lead_ind,tstep_count,year_count] = \
                                                        curr_outputs[1][n_ind,year_count]
                            postinit_t_ind_alltstep[lead_ind,tstep_count,year_count] = \
                                                        curr_outputs[2][n_ind,year_count]
                            postinit_lead_ind_alltstep[lead_ind,tstep_count,year_count] = lead_ind
                            postinit_init_times_array[lead_ind,pred_count,tstep_count,year_count] = \
                                                        forc_postinit_init_time[curr_outputs[1][n_ind,year_count]]
                if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                    curr_outputs = postinit_forcing_index_arrays(inputs,sens_shape0,\
                                                                 closest_forc_to_sens_atyears_currmo,\
                                                                 forc_postinit_dict['postinit_subt'])
                    for year_count in range(len(curr_outputs[0])):
                        postinit_subt_good_years_ind_alltstep[:,tstep_count,year_count] = \
                                                                curr_outputs[0][year_count]
                        for n_ind,lead_ind in enumerate(curr_outputs[3][:,year_count]):
                            if lead_ind >= 0:
                                postinit_subt_init_ind_alltstep[lead_ind,tstep_count,year_count] = \
                                                                    curr_outputs[1][n_ind,year_count]
                                postinit_subt_t_ind_alltstep[lead_ind,tstep_count,year_count] = \
                                                                    curr_outputs[2][n_ind,year_count]
                                postinit_subt_lead_ind_alltstep[lead_ind,tstep_count,year_count] = lead_ind
        
        
        
        # # compute forcing means across years by looping through forcing time chunks 
        # # and adj time steps
        
        n_t_chunks = len(inputs['years_to_predict'])
        
        preinit_t_chunksize = np.int64(np.ceil(forc_preinit_np_array.shape[0]/n_t_chunks))
        cumsum_preinit_mask_forc = np.zeros(sens.shape)
        cumsum_preinit_mask = np.zeros(sens.shape,dtype=np.float64)
        if add_runoff:
            preinit_r_t_chunksize = np.int64(np.ceil(forc_preinit_r_np_array.shape[0]/n_t_chunks))
            cumsum_preinit_r_mask_forc = np.zeros(sens.shape)
            cumsum_preinit_r_mask = np.zeros(sens.shape,dtype=np.float64)            
        if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
            preinit_subt_t_chunksize = np.int64(np.ceil(forc_preinit_subtract_np_array.shape[0]/n_t_chunks))
            cumsum_preinit_subt_mask_forc = np.zeros(sens.shape)
            cumsum_preinit_subt_mask = np.zeros(sens.shape,dtype=np.float64)
        if 'forcing_postinit' in inputs.keys():
            postinit_impact_sum_attstep = np.zeros((n_postinit_leads,sens_shape0)\
                  +forc_postinit_array.shape[-3:])
            postinit_t_chunksize = np.int64(np.ceil(forc_postinit_array.shape[0]/n_t_chunks))
            cumsum_postinit_mask_forc = np.zeros((n_postinit_leads,)+sens.shape)
            cumsum_postinit_mask = np.zeros((n_postinit_leads,)+sens.shape,dtype=np.float64)
            if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                postinit_subt_impact_sum_attstep = np.empty((n_postinit_leads,sens_shape0)\
                      +forc_postinit_subtract_array.shape[-3:])
                postinit_subt_t_chunksize = np.int64(np.ceil(forc_postinit_subtract_array.shape[0]/n_t_chunks))
                cumsum_postinit_subt_mask_forc = np.zeros((n_postinit_subt_leads,)+sens.shape)
                cumsum_postinit_subt_mask = np.zeros((n_postinit_subt_leads,)+sens.shape,dtype=np.float64)
        
        for t_chunk in range(n_t_chunks):
            pre_chunk_start = preinit_t_chunksize*t_chunk
            pre_chunk_end = np.fmin(preinit_t_chunksize*(t_chunk+1),forc_preinit_np_array.shape[0])
            pre_chunk_array = forc_preinit_np_array[pre_chunk_start:pre_chunk_end,:,:,:]
            if add_runoff:
                pre_r_chunk_start = preinit_r_t_chunksize*t_chunk
                pre_r_chunk_end = np.fmin(preinit_r_t_chunksize*(t_chunk+1),forc_preinit_r_np_array.shape[0])
                pre_r_chunk_array = forc_preinit_r_np_array[pre_r_chunk_start:pre_r_chunk_end,:,:,:]                
            if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                presub_chunk_start = preinit_subt_t_chunksize*t_chunk
                presub_chunk_end = np.fmin(preinit_subt_t_chunksize*(t_chunk+1),forc_preinit_subtract_np_array.shape[0])
                presub_chunk_array = forc_preinit_subtract_np_array[presub_chunk_start:presub_chunk_end,:,:,:]             
            if 'forcing_postinit' in inputs.keys():
                post_chunk_start = postinit_t_chunksize*t_chunk
                post_chunk_end = np.fmin(postinit_t_chunksize*(t_chunk+1),forc_postinit_array.shape[0])
                post_chunk_array = forc_postinit_array[post_chunk_start:post_chunk_end,:,:,:,:].values
                if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                    postsub_chunk_start = postinit_subt_t_chunksize*t_chunk
                    postsub_chunk_end = np.fmin(postinit_subt_t_chunksize*(t_chunk+1),\
                                                forc_postinit_subtract_array.shape[0])
                    postsub_chunk_array = forc_postinit_subtract_array[postsub_chunk_start:postsub_chunk_end,:,:,:,:]\
                                            .values
            
            for tstep_count in range(sens_shape0):
                sens_tstep = sens[tstep_count,:,:,:]
                
                cumsum_preinit_mask_forc_attstep = cumsum_preinit_mask_forc[tstep_count,:,:,:]
                cumsum_preinit_mask_attstep = cumsum_preinit_mask[tstep_count,:,:,:]
                cumsum_preinit_mask_forc_attstep,cumsum_preinit_mask_attstep = \
                            forc_cumsum_in_t_range(pre_chunk_array,pre_chunk_start,pre_chunk_end,\
                                                       preinit_t_ind_alltstep[tstep_count,:],\
                                                       cumsum_preinit_mask_forc_attstep,\
                                                       cumsum_preinit_mask_attstep)
                if add_runoff:
                    cumsum_preinit_r_mask_forc_attstep = cumsum_preinit_r_mask_forc[tstep_count,:,:,:]
                    cumsum_preinit_r_mask_attstep = cumsum_preinit_r_mask[tstep_count,:,:,:]
                    cumsum_preinit_r_mask_forc_attstep,cumsum_preinit_r_mask_attstep = \
                                forc_cumsum_in_t_range(pre_r_chunk_array,pre_r_chunk_start,pre_r_chunk_end,\
                                                           preinit_r_t_ind_alltstep[tstep_count,:],\
                                                           cumsum_preinit_r_mask_forc_attstep,\
                                                           cumsum_preinit_r_mask_attstep)

                if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                    cumsum_preinit_subt_mask_forc_attstep = cumsum_preinit_subt_mask_forc[tstep_count,:,:,:]
                    cumsum_preinit_subt_mask_attstep = cumsum_preinit_subt_mask[tstep_count,:,:,:]
                    cumsum_preinit_subt_mask_forc_attstep,cumsum_preinit_subt_mask_attstep = \
                                forc_cumsum_in_t_range(presub_chunk_array,presub_chunk_start,presub_chunk_end,\
                                                           preinit_subt_t_ind_alltstep[tstep_count,:],\
                                                           cumsum_preinit_subt_mask_forc_attstep,\
                                                           cumsum_preinit_subt_mask_attstep)

                if 'forcing_postinit' in inputs.keys():
                    cumsum_postinit_mask_forc_attstep = cumsum_postinit_mask_forc[:,tstep_count,:,:,:]
                    cumsum_postinit_mask_attstep = cumsum_postinit_mask[:,tstep_count,:,:,:]
                    cumsum_postinit_mask_forc_attstep,cumsum_postinit_mask_attstep = \
                            forc_cumsum_in_t_range_postinit(post_chunk_array,post_chunk_start,post_chunk_end,\
                                                               postinit_init_ind_alltstep[:,tstep_count,:],\
                                                               postinit_t_ind_alltstep[:,tstep_count,:],\
                                                               postinit_lead_ind_alltstep[:,tstep_count,:],\
                                                               cumsum_postinit_mask_forc_attstep,\
                                                               cumsum_postinit_mask_attstep)
                        
        
                    if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                        cumsum_postinit_subt_mask_forc_attstep = \
                                        cumsum_postinit_subt_mask_forc[:,tstep_count,:,:,:]
                        cumsum_postinit_mask_subt_attstep = cumsum_postinit_subt_mask[:,tstep_count,:,:,:]
                        cumsum_postinit_subt_mask_forc_attstep,cumsum_postinit_subt_mask_attstep = \
                          forc_cumsum_in_t_range_postinit(postsub_chunk_array,postsub_chunk_start,postsub_chunk_end,\
                                                               postinit_subt_init_ind_alltstep[:,tstep_count,:],\
                                                               postinit_subt_t_ind_alltstep[:,tstep_count,:],\
                                                               postinit_subt_lead_ind_alltstep[:,tstep_count,:],\
                                                               cumsum_postinit_subt_mask_forc_attstep,\
                                                               cumsum_postinit_subt_mask_attstep)
            
        # compute means across years
        preinit_forc_tmean = cumsum_preinit_mask_forc/cumsum_preinit_mask
        if add_runoff:
            preinit_r_forc_tmean = cumsum_preinit_r_mask_forc/cumsum_preinit_r_mask
        if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
            preinit_subt_forc_tmean = cumsum_preinit_subt_mask_forc/cumsum_preinit_subt_mask
        if 'forcing_postinit' in inputs.keys():
            postinit_forc_tmean = cumsum_postinit_mask_forc/cumsum_postinit_mask
            if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                postinit_subt_forc_tmean = cumsum_postinit_subt_mask_forc/cumsum_postinit_subt_mask

        
        
        # # compute cell impacts by looping through forcing time chunks and adj time steps
        # # (with means removed)
        
        for t_chunk in range(n_t_chunks):
            pre_chunk_start = preinit_t_chunksize*t_chunk
            pre_chunk_end = np.fmin(preinit_t_chunksize*(t_chunk+1),forc_preinit_np_array.shape[0])
            pre_chunk_array = forc_preinit_np_array[pre_chunk_start:pre_chunk_end,:,:,:]
            if add_runoff:
                pre_r_chunk_start = preinit_r_t_chunksize*t_chunk
                pre_r_chunk_end = np.fmin(preinit_r_t_chunksize*(t_chunk+1),forc_preinit_r_np_array.shape[0])
                pre_r_chunk_array = forc_preinit_r_np_array[pre_r_chunk_start:pre_r_chunk_end,:,:,:]                
            if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                presub_chunk_start = preinit_subt_t_chunksize*t_chunk
                presub_chunk_end = np.fmin(preinit_subt_t_chunksize*(t_chunk+1),forc_preinit_subtract_np_array.shape[0])
                presub_chunk_array = forc_preinit_subtract_np_array[presub_chunk_start:presub_chunk_end,:,:,:]             
            if 'forcing_postinit' in inputs.keys():
                post_chunk_start = postinit_t_chunksize*t_chunk
                post_chunk_end = np.fmin(postinit_t_chunksize*(t_chunk+1),forc_postinit_array.shape[0])
                post_chunk_array = forc_postinit_array[post_chunk_start:post_chunk_end,:,:,:,:].values
                if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                    postsub_chunk_start = postinit_subt_t_chunksize*t_chunk
                    postsub_chunk_end = np.fmin(postinit_subt_t_chunksize*(t_chunk+1),\
                                                forc_postinit_subtract_array.shape[0])
                    postsub_chunk_array = forc_postinit_subtract_array[postsub_chunk_start:postsub_chunk_end,:,:,:,:]\
                                            .values
            
            for tstep_count in range(sens_shape0):
                sens_tstep = sens[tstep_count,:,:,:]

                sensadj_opt_ratios_dict = {}
                curr_arrays_dict = {'preinit':pre_chunk_array}
                curr_tmean_attstep_dict = {'preinit':preinit_forc_tmean[tstep_count,:,:,:]}
                chunk_bounds_dict = {'preinit':[pre_chunk_start,pre_chunk_end]}
                curr_cell_impact_attstep_dict = {'preinit':preinit_cell_impact_attimes[tstep_count,:,:]}
                good_years_ind_attstep_dict = {'preinit':preinit_good_years_ind_alltstep[tstep_count,:]}
                t_ind_attstep_dict = {'preinit':preinit_t_ind_alltstep[tstep_count,:]}
                if add_runoff:
                    curr_arrays_dict['preinit_runoff'] = pre_r_chunk_array
                    curr_tmean_attstep_dict['preinit_runoff'] = preinit_r_forc_tmean[tstep_count,:,:,:]
                    chunk_bounds_dict['preinit_runoff'] = [pre_r_chunk_start,pre_r_chunk_end]
                    curr_cell_impact_attstep_dict['preinit_runoff'] = preinit_r_cell_impact_attimes[tstep_count,:,:]
                    good_years_ind_attstep_dict['preinit_runoff'] = preinit_r_good_years_ind_alltstep[tstep_count,:]
                    t_ind_attstep_dict['preinit_runoff'] = preinit_r_t_ind_alltstep[tstep_count,:]
                if 'sensadj' in inputs['forcing_preinit']['forc_type']:
                    curr_forc_num = (sensadj_forc_nums == curr_forc).nonzero()[0][0]
                    sensadj_opt_ratios_dict['preinit'] = [True,sensadj_ratios[curr_forc_num,pred_count,tstep_count]]
                else:
                    sensadj_opt_ratios_dict['preinit'] = []
                if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                    curr_arrays_dict['preinit_subt'] = presub_chunk_array
                    curr_tmean_attstep_dict['preinit_subt'] = preinit_subt_forc_tmean[tstep_count,:,:,:]
                    chunk_bounds_dict['preinit_subt'] = [presub_chunk_start,presub_chunk_end]
                    curr_cell_impact_attstep_dict['preinit_subt'] = preinit_subt_cell_impact_attimes[tstep_count,:,:]
                    if 'sensadj' in inputs['forcing_preinit']['forc_subtract_type']:
                        curr_forc_num = (sensadj_forc_nums == curr_forc).nonzero()[0][0]
                        sensadj_opt_ratios_dict['preinit_subt'] = \
                                                    [True,sensadj_ratios[curr_forc_num,pred_count,tstep_count]]
                    else:
                        sensadj_opt_ratios_dict['preinit_subt'] = []
                    good_years_ind_attstep_dict['preinit_subt'] = preinit_subt_good_years_ind_alltstep[tstep_count,:]
                    t_ind_attstep_dict['preinit_subt'] = preinit_subt_t_ind_alltstep[tstep_count,:]
                if 'forcing_postinit' in inputs.keys():
                    curr_arrays_dict['postinit'] = post_chunk_array
                    curr_tmean_attstep_dict['postinit'] = postinit_forc_tmean[:,tstep_count,:,:,:]
                    chunk_bounds_dict['postinit'] = [post_chunk_start,post_chunk_end]
                    curr_cell_impact_attstep_dict['postinit'] = postinit_cell_impact_attimes[:,tstep_count,:,:]
                    if 'sensadj' in inputs['forcing_postinit']['forc_type']:
                        curr_forc_num = (sensadj_forc_nums == curr_forc).nonzero()[0][0]
                        sensadj_opt_ratios_dict['postinit'] = [True,sensadj_ratios[curr_forc_num,pred_count,tstep_count]]
                    else:
                        sensadj_opt_ratios_dict['postinit'] = []
                    good_years_ind_attstep_dict['postinit'] = postinit_good_years_ind_alltstep[:,tstep_count,:]
                    t_ind_attstep_dict['postinit_init'] = postinit_init_ind_alltstep[:,tstep_count,:]
                    t_ind_attstep_dict['postinit_t'] = postinit_t_ind_alltstep[:,tstep_count,:]
                    t_ind_attstep_dict['postinit_lead'] = postinit_lead_ind_alltstep[:,tstep_count,:]
                    if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                        curr_arrays_dict['postinit_subt'] = postsub_chunk_array
                        curr_tmean_attstep_dict['postinit_subt'] = postinit_subt_forc_tmean[:,tstep_count,:,:,:]
                        chunk_bounds_dict['postinit_subt'] = [postsub_chunk_start,postsub_chunk_end]
                        curr_cell_impact_attstep_dict['postinit_subt'] = \
                                                        postinit_subt_cell_impact_attimes[:,tstep_count,:,:]
                        if 'sensadj' in inputs['forcing_postinit']['forc_subtract_type']:
                            curr_forc_num = (sensadj_forc_nums == curr_forc).nonzero()[0][0]
                            sensadj_opt_ratios_dict['postinit_subt'] = \
                                                        [True,sensadj_ratios[curr_forc_num,pred_count,tstep_count]]
                        else:
                            sensadj_opt_ratios_dict['postinit_subt'] = []
                        good_years_ind_attstep_dict['postinit_subt'] = \
                                                        postinit_subt_good_years_ind_alltstep[:,tstep_count,:]
                        t_ind_attstep_dict['postinit_subt_init'] = postinit_subt_init_ind_alltstep[:,tstep_count,:]
                        t_ind_attstep_dict['postinit_subt_t'] = postinit_subt_t_ind_alltstep[:,tstep_count,:]
                        t_ind_attstep_dict['postinit_subt_lead'] = postinit_subt_lead_ind_alltstep[:,tstep_count,:]

                curr_cell_impact_arrays = cell_impacts_aggregate_allforc(\
                                                 curr_arrays_dict,curr_tmean_attstep_dict,\
                                                 chunk_bounds_dict,\
                                                 curr_cell_impact_attstep_dict,\
                                                 sens_tstep,sensadj_opt_ratios_dict,\
                                                 good_years_ind_attstep_dict,t_ind_attstep_dict,\
                                                 inputs['cells_all_single'],cell_indices,\
                                                 cell_indices_np_list,tstep_count)

                preinit_cell_impact_attimes[tstep_count,:,:] = curr_cell_impact_arrays['preinit']
                if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                    preinit_subt_cell_impact_attimes[tstep_count,:,:] = curr_cell_impact_arrays['preinit_subt']
                if 'forcing_postinit' in inputs.keys():
                    postinit_cell_impact_attimes[:,tstep_count,:,:] = curr_cell_impact_arrays['postinit']
                    if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                        postinit_subt_cell_impact_attimes[:,tstep_count,:,:] = curr_cell_impact_arrays['postinit_subt']    
            
            
        # subtract cell impact arrays if we do not need them separated later
        if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
            if resid_compute:
                preinit_cell_impact_attimes -= preinit_subt_cell_impact_attimes
        if 'forcing_postinit' in inputs.keys():
            if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                if resid_compute:
                    postinit_cell_impact_attimes -= postinit_subt_cell_impact_attimes

        
        # # assemble cumulative impact arrays
        
        for tstep_count in range(sens_shape0):            
            if tstep_count == 0:
                cum_preinit_cell_impact_attimes[tstep_count,:,:] = preinit_cell_impact_attimes[tstep_count,:,:]
            else:
                cum_preinit_cell_impact_attimes[tstep_count,:,:] = \
                  cum_preinit_cell_impact_attimes[tstep_count-1,:,:] + preinit_cell_impact_attimes[tstep_count,:,:]
            
            # # aggregate cumulative impacts from preinit "subtracted" forcing
            # # (only if resid var will be computed and subtracted from preinit forcing)
            if resid_compute:
                if 'forc_subtract_type' in inputs['forcing_preinit'].keys():            
                    if tstep_count == 0:
                        cum_preinit_subt_cell_impact_attimes[tstep_count,:,:] = \
                          preinit_subt_cell_impact_attimes[tstep_count,:,:]
                    else:
                        cum_preinit_subt_cell_impact_attimes[tstep_count,:,:] = \
                          cum_preinit_subt_cell_impact_attimes[tstep_count-1,:,:] + \
                          preinit_subt_cell_impact_attimes[tstep_count,:,:]
            
            
            if 'forcing_postinit' in inputs.keys():
                if tstep_count > 0:
                    init_times_unique = np.unique(postinit_init_times_array[:,pred_count,:tstep_count,:]\
                                                  .astype('datetime64[D]'))
                    init_times_unique = init_times_unique[~np.isnan(init_times_unique)]
                    for curr_init_time in init_times_unique:
                        init_time_mask = (np.abs(postinit_init_times_array[:,pred_count,:tstep_count,:]\
                                                  - curr_init_time) < np.timedelta64(2,'D')).astype('int32')
                        init_time_mask_attstep = (np.abs(postinit_init_times_array[:,pred_count,tstep_count,:]\
                                                          - curr_init_time) < np.timedelta64(2,'D'))\
                                                            .astype('int32')
                        cum_postinit_cell_impact_attimes[:,:tstep_count,:,:] += \
                            np.expand_dims(init_time_mask,axis=-1)\
                                   *np.nansum(np.expand_dims(init_time_mask_attstep,axis=-1)\
                                              *postinit_cell_impact_attimes[:,tstep_count,:,:],\
                                              axis=0,keepdims=True)
                    
                if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                    if resid_compute:
                        postinit_subt_cell_impact_attimes_atforc[:,pred_count,tstep_count,:,:] = \
                          postinit_subt_cell_impact_attimes[:,tstep_count,:,:]
                        if tstep_count > 0:
                            # here it is assumed the subt array has the same postinit_init_times array as the
                            # primary postinit array
                            cum_postinit_subt_cell_impact_attimes[:,:tstep_count,:,:] += \
                                np.expand_dims(init_time_mask,axis=-1)\
                                       *np.nansum(np.expand_dims(init_time_mask_attstep,axis=-1)\
                                                  *postinit_subt_cell_impact_attimes[:,tstep_count,:,:],\
                                                  axis=0,keepdims=True)

        # replace NaN padding for preinit cumulative arrays at the end of some prediction months
        cum_preinit_cell_impact_attimes[-1,:,:] = np.where(~np.isnan(cum_preinit_cell_impact_attimes[-1,:,:]),\
                                                           cum_preinit_cell_impact_attimes[-1,:,:],\
                                                           cum_preinit_cell_impact_attimes[-2,:,:])
        if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
            cum_preinit_subt_cell_impact_attimes[-1,:,:] = \
                                            np.where(~np.isnan(cum_preinit_subt_cell_impact_attimes[-1,:,:]),\
                                                       cum_preinit_subt_cell_impact_attimes[-1,:,:],\
                                                       cum_preinit_subt_cell_impact_attimes[-2,:,:])

        
        
        # # compute residuals as needed

        if 'forcing_postinit' in inputs.keys():
            # mask to determine whether to use preinit or postinit forcing
            postinit_mask = (np.expand_dims(closest_forc_to_sens_atyears_currmo,axis=0)\
                 - postinit_init_times_array[:,pred_count,-len_tsteps_dim:,:]\
                 > np.timedelta64(0,'ns')).astype('float64')
            postinit_mask_expand = np.expand_dims(postinit_mask,axis=-1)
        
        if resid_compute:
            # load reference time series in current prediction month
            if 'obs' in inputs.keys():
                ref_datetime = inputs['obs']['datetime']
                ref_tseries = inputs['obs']['tseries']
            elif 'reconstr' in inputs.keys():
                ref_datetime = inputs['reconstr']['datetime']
                ref_tseries = inputs['reconstr']['tseries']
            elif 'model' in inputs.keys():
                ref_datetime = inputs['model']['datetime']
                ref_tseries = inputs['model']['tseries']
            if len(ref_tseries.shape) > 1:
                ref_tseries_inmonth = np.empty((ref_tseries.shape[0],len(years_to_predict)))
            else:
                ref_tseries_inmonth = np.empty((len(years_to_predict),))
            ref_tseries_inmonth.fill(np.nan)
            for year_count,year in enumerate(years_to_predict):
                curr_pred_time = np.datetime64(str(year)+'-'+str(pred_month).rjust(2,'0')+'-15','ns')
                closest_ref_t = np.nanargmin(np.abs(ref_datetime - curr_pred_time))
                if np.abs(ref_datetime[closest_ref_t] - curr_pred_time) < np.timedelta64(10,'D'):
                    if len(ref_tseries.shape) > 1:
                        ref_tseries_inmonth[:,year_count] = ref_tseries[:,closest_ref_t]
                    else:
                        ref_tseries_inmonth[year_count] = ref_tseries[closest_ref_t]
                
            # remove means and compute residual time series for current prediction month
            mask_obs = nanmask_create(ref_tseries_inmonth)
            ref_tseries_inmonth -= (np.nansum(mask_obs*ref_tseries_inmonth)/np.nansum(mask_obs))
            if trend_remove:
                ref_tseries_inmonth -= trend_compute_dimvec(\
                                        ref_tseries_inmonth,\
                    inputs['years_to_predict'],axis=-1,output_trendline=True)[1]
            
            if sensadj_compute == True:
                curr_obs_weight = inputs['sensadj_params']['obs_weights_array'][curr_forc_num,pred_count,:]
                curr_obs_part = np.expand_dims(curr_obs_weight,axis=-1)*np.expand_dims(ref_tseries_inmonth,axis=0)
                curr_resid_inpart = curr_obs_part - np.nansum(\
                  preinit_cell_impact_attimes,axis=-1)
                if trend_remove == True:
                    curr_resid_inpart -= trend_compute_dimvec(\
                                          curr_resid_inpart,\
                     inputs['years_to_predict'],axis=1,output_trendline=True)[1]
                resid_var_inpart[pred_count,:curr_resid_inpart.shape[0]] = var_compute(curr_resid_inpart,\
                                                                                      axis=-1)
            
            
            # # compute residual variances
            if 'forcing_postinit' in inputs.keys():
                if len(ref_tseries_inmonth.shape) < 2:
                    ref_tseries_inmonth = np.expand_dims(ref_tseries_inmonth,axis=0)
                resid_tseries_inmonth = np.expand_dims(ref_tseries_inmonth,axis=(1,3))\
                  - (((1 - postinit_mask_expand)*np.expand_dims(preinit_cell_impact_attimes,axis=0))\
                     + (postinit_mask_expand*postinit_cell_impact_attimes))
                cum_resid_tseries_inmonth = np.expand_dims(ref_tseries_inmonth,axis=(1,3))\
                  - (np.expand_dims(cum_preinit_cell_impact_attimes,axis=0) + cum_postinit_cell_impact_attimes)
                
                if ('forc_subtract_type' in inputs['forcing_preinit'].keys()) or \
                          ('forc_subtract_type' in inputs['forcing_postinit'].keys()):
                    resid_tseries_subt_inmonth = np.tile(np.expand_dims(ref_tseries_inmonth,axis=(1,3)),\
                      (n_postinit_leads,len_tsteps_dim,1,n_cells))
                    cum_resid_tseries_subt_inmonth = np.tile(np.expand_dims(ref_tseries_inmonth,axis=(1,3)),\
                      (n_postinit_leads,len_tsteps_dim,1,n_cells))
                if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                    resid_tseries_subt_inmonth -= preinit_subt_cell_impact_attimes
                    cum_resid_tseries_subt_inmonth -= cum_preinit_subt_cell_impact_attimes
                if 'forc_subtract_type' in inputs['forcing_postinit'].keys():
                    resid_tseries_subt_inmonth -= postinit_subt_cell_impact_attimes
                    cum_resid_tseries_subt_inmonth -= cum_postinit_subt_cell_impact_attimes
            else:
                resid_tseries_inmonth = np.expand_dims(ref_tseries_inmonth,axis=(0,2))\
                  - preinit_cell_impact_attimes
                cum_resid_tseries_inmonth = np.expand_dims(ref_tseries_inmonth,axis=(0,2))\
                  - cum_preinit_cell_impact_attimes
                if 'forc_subtract_type' in inputs['forcing_preinit'].keys():
                    resid_tseries_subt_inmonth = np.expand_dims(ref_tseries_inmonth,axis=(0,2))\
                      - preinit_subt_cell_impact_attimes
                    cum_resid_tseries_subt_inmonth = np.expand_dims(ref_tseries_inmonth,axis=(0,2))\
                      - cum_preinit_subt_cell_impact_attimes
            
            if trend_remove == True:
                resid_tseries_inmonth -= trend_compute_dimvec(\
                  resid_tseries_inmonth,\
                  inputs['years_to_predict'],axis=-2,output_trendline=True)[1]
                cum_resid_tseries_inmonth -= trend_compute_dimvec(\
                  cum_resid_tseries_inmonth,\
                  inputs['years_to_predict'],axis=-2,output_trendline=True)[1]
            if 'forcing_postinit' in inputs.keys():
                resid_cell_impact_var_atforc[:,pred_count,-len_tsteps_dim:,:] = \
                                                                var_compute(resid_tseries_inmonth,axis=-2)
                cum_resid_cell_impact_var_atforc[:,pred_count,-len_tsteps_dim:,:] = \
                                                                var_compute(cum_resid_tseries_inmonth,axis=-2)
            else:
                resid_cell_impact_var_atforc[pred_count,-len_tsteps_dim:,:] = \
                                                                var_compute(resid_tseries_inmonth,axis=-2)
                cum_resid_cell_impact_var_atforc[pred_count,-len_tsteps_dim:,:] = \
                                                                var_compute(cum_resid_tseries_inmonth,axis=-2)
            if 'resid_tseries_subt_inmonth' in dir():
                # subtract residual variances from "subtract" forcings
                if trend_remove == True:
                    resid_tseries_subt_inmonth -= trend_compute_dimvec(\
                      resid_tseries_subt_inmonth,\
                      inputs['years_to_predict'],axis=-2,output_trendline=True)[1]
                    cum_resid_tseries_subt_inmonth -= trend_compute_dimvec(\
                      cum_resid_tseries_subt_inmonth,\
                      inputs['years_to_predict'],axis=-2,output_trendline=True)[1]                
                resid_subt_cell_impact_var = var_compute(resid_tseries_subt_inmonth,axis=-2)
                cum_resid_subt_cell_impact_var = var_compute(cum_resid_tseries_subt_inmonth,axis=-2)
                if 'forcing_postinit' in inputs.keys():
                    resid_cell_impact_var_atforc[:,pred_count,-len_tsteps_dim:,:] -= \
                                                                                resid_subt_cell_impact_var
                    cum_resid_cell_impact_var_atforc[:,pred_count,-len_tsteps_dim:,:] -= \
                                                                                cum_resid_subt_cell_impact_var
                else:
                    resid_cell_impact_var_atforc[pred_count,-len_tsteps_dim:,:] -= \
                                                                                resid_subt_cell_impact_var
                    cum_resid_cell_impact_var_atforc[pred_count,-len_tsteps_dim:,:] -= \
                                                                                cum_resid_subt_cell_impact_var
        else:
            # # compute impact variances
            if 'forcing_postinit' in inputs.keys():
                impact_tseries_inmonth = ((1 - postinit_mask_expand)*np.expand_dims(preinit_cell_impact_attimes,axis=0))\
                  + (postinit_mask_expand*postinit_cell_impact_attimes)
                cum_impact_tseries_inmonth = cum_preinit_cell_impact_attimes + cum_postinit_cell_impact_attimes
            else:
                impact_tseries_inmonth = preinit_cell_impact_attimes
                cum_impact_tseries_inmonth = cum_preinit_cell_impact_attimes
            if trend_remove == True:
                impact_tseries_inmonth -= trend_compute_dimvec(\
                  impact_tseries_inmonth,\
                  inputs['years_to_predict'],axis=-2,output_trendline=True)[1]
                cum_impact_tseries_inmonth -= trend_compute_dimvec(\
                  cum_impact_tseries_inmonth,\
                  inputs['years_to_predict'],axis=-2,output_trendline=True)[1]
            if 'forcing_postinit' in inputs.keys():
                cell_impact_var_atforc[:,pred_count,-len_tsteps_dim:,:] = \
                                                                var_compute(impact_tseries_inmonth,axis=-2)
                cum_cell_impact_var_atforc[:,pred_count,-len_tsteps_dim:,:] = \
                                                                var_compute(cum_impact_tseries_inmonth,axis=-2)
            else:
                cell_impact_var_atforc[pred_count,-len_tsteps_dim:,:] = \
                                                                var_compute(impact_tseries_inmonth,axis=-2)
                cum_cell_impact_var_atforc[pred_count,-len_tsteps_dim:,:] = \
                                                                var_compute(cum_impact_tseries_inmonth,axis=-2)

        sens_times[pred_count,-len_tsteps_dim:] = sens_times_currmo
        closest_forc_to_sens_atyears\
          [pred_count,-len_tsteps_dim:,:] = \
          closest_forc_to_sens_atyears_currmo

        preinit_impact_senststep_attimes[pred_count,-len_tsteps_dim:,:] = \
                                                                np.nansum(preinit_cell_impact_attimes,axis=-1)
        if 'forcing_postinit' in inputs.keys():
            postinit_impact_senststep_attimes[:,pred_count,-len_tsteps_dim:,:] = \
                                                                np.nansum(postinit_cell_impact_attimes,axis=-1)
        if inputs['cell_impact_years_save']:
            if trend_remove:
                preinit_cell_impact_attimes -= trend_compute_dimvec(\
                                    preinit_cell_impact_attimes,\
                                    inputs['years_to_predict'],axis=-2,output_trendline=True)[1]
            preinit_cell_impact_attimes_atforc[pred_count,-len_tsteps_dim:,:,:] = \
                                                                preinit_cell_impact_attimes
            if 'forcing_postinit' in inputs.keys():
                if trend_remove:
                    postinit_cell_impact_attimes -= trend_compute_dimvec(\
                                        postinit_cell_impact_attimes,\
                                        inputs['years_to_predict'],axis=-2,output_trendline=True)[1]
                postinit_cell_impact_attimes_atforc[:,pred_count,-len_tsteps_dim:,:,:] = \
                                                                postinit_cell_impact_attimes
        
#         time_log.append(time.time())
#         import pdb
#         pdb.set_trace()
    
    
    if 'forcing_postinit' in inputs.keys():        
        if (('forc_subtract_type' in inputs['forcing_postinit'].keys()) and resid_compute):
            postinit_cell_impact_attimes_atforc -= postinit_subt_cell_impact_attimes_atforc
    if (('forc_subtract_type' in inputs['forcing_preinit'].keys()) and resid_compute):
        preinit_cell_impact_attimes_atforc -= preinit_subt_cell_impact_attimes_atforc

    if 'forcing_postinit' not in inputs.keys():
        postinit_init_times_array = np.array([])
    
    if inputs['cell_impact_years_save'] == True:
        cell_impact_attimes_atforc = {'preinit':preinit_cell_impact_attimes_atforc}
        if 'forcing_postinit' in inputs.keys():
            cell_impact_attimes_atforc['postinit'] = postinit_cell_impact_attimes_atforc
        
        if resid_compute:
            if sensadj_compute == True:
                return cell_impact_attimes_atforc,resid_cell_impact_var_atforc,cum_resid_cell_impact_var_atforc,resid_var_inpart,\
                  postinit_init_times_array,sens_times,closest_forc_to_sens_atyears
            else:
                return cell_impact_attimes_atforc,resid_cell_impact_var_atforc,cum_resid_cell_impact_var_atforc,\
                  postinit_init_times_array,sens_times,closest_forc_to_sens_atyears
        else:
            return cell_impact_attimes_atforc,cell_impact_var_atforc,cum_cell_impact_var_atforc,\
              postinit_init_times_array,sens_times,closest_forc_to_sens_atyears
    else:
        impact_senststep_attimes = {'preinit':preinit_impact_senststep_attimes}
        if 'forcing_postinit' in inputs.keys():
            impact_senststep_attimes['postinit'] = postinit_impact_senststep_attimes
        
        if resid_compute:
            return impact_senststep_attimes,resid_cell_impact_var_atforc,cum_resid_cell_impact_var_atforc,\
              postinit_init_times_array,sens_times,closest_forc_to_sens_atyears
        else:
            return impact_senststep_attimes,cell_impact_var_atforc,cum_cell_impact_var_atforc,\
              postinit_init_times_array,sens_times,closest_forc_to_sens_atyears