"""Functions to load and pre-process reference time series 
   (for computing residuals and variances explained)."""

import numpy as np
import xarray as xr
import sys
from os.path import join,expanduser,exists
import glob

user_home_dir = expanduser('~')
sys.path.append(join(user_home_dir,'Documents','py_functions'))
from filter_functions import *



def nanmask_create(array):
    mask = np.logical_and(np.logical_and(~np.isnan(array),~np.isinf(array)),np.abs(array) >= 1.e-15)\
      .astype('int64')
    return mask


def tgauge_mon_read(tgauge_obs_filename):
    tval = []
    sl = []
    with open(tgauge_obs_filename,'r') as tgauge_file:
        curr_lines = tgauge_file.readlines()
        for curr_line in curr_lines:
            curr_items = curr_line.split(";")
            tval.append(float(curr_items[0]))
            sl.append(float(curr_items[1]))

    tval_array = np.asarray(tval).astype('float32')
    year_array = np.floor(tval_array).astype('int64')
    month_array = np.ceil(12*(tval_array - year_array)).astype('int64')
    datetime_array = (year_array - 1970).astype('datetime64[Y]') + ((month_array - 1).astype('timedelta64[M]'))\
                        + np.timedelta64(14,'D')
    sl_array = np.asarray(sl).astype('float32')
    sl_array[sl_array < -50000] = np.nan
    sl_array = 0.001*sl_array
    
    return sl_array,datetime_array


def tgauge_IB_correction(tgauge_sl,tgauge_datetime,lat_pt,lon_pt,ds_ERA5_mon_slp):
    
    rho = 1030
    g = 9.81
    
    ERA5_closest_lat_ind = np.argmin(np.abs(ds_ERA5_mon_slp.lat.values - lat_pt))
    ERA5_closest_lon_ind = np.argmin(np.abs(((ds_ERA5_mon_slp.lon.values - lon_pt + 180) % 360) - 180))
    if ds_ERA5_mon_slp.slm.isel(lat=ERA5_closest_lat_ind,lon=ERA5_closest_lon_ind).values < 0.5:
        lat_array = np.tile(np.reshape(ds_ERA5_mon_slp.lat.values,(-1,1)),(1,ds_ERA5_mon_slp.sizes['lon']))
        lon_array = np.tile(np.reshape(ds_ERA5_mon_slp.lon.values,(1,-1)),(ds_ERA5_mon_slp.sizes['lat'],1))
        dist_from_pt = np.abs(111100*((np.cos((np.pi/180)*lat_pt)*(((lon_array - lon_pt + 180) % 360) - 180))\
                                      + (1j*(lat_array - lat_pt))))
        dist_from_pt = np.where(ds_ERA5_mon_slp.slm.values > 0.5,dist_from_pt,np.nan)
        closest_good_pt = np.unravel_index(np.nanargmin(dist_from_pt),ds_ERA5_mon_slp.slm.shape)
        ERA5_closest_lat_ind,ERA5_closest_lon_ind = closest_good_pt
    mon_slp_closest = ds_ERA5_mon_slp.msl.isel(lat=ERA5_closest_lat_ind,lon=ERA5_closest_lon_ind).values
    mon_slp_minus_ocemean = mon_slp_closest - ds_ERA5_mon_slp.mslp_ocean.values
    ERA5_year_array = np.floor(ds_ERA5_mon_slp.tval.values)
    ERA5_month_array = np.ceil(12*(ds_ERA5_mon_slp.tval.values - ERA5_year_array)).astype('int64')
    ERA5_datetimes = (ERA5_year_array - 1970).astype('datetime64[Y]')\
                        + ((ERA5_month_array - 1).astype('timedelta64[M]'))\
                        + np.timedelta64(14,'D')
    tgauge_sl_corrected = np.empty(tgauge_sl.shape)
    tgauge_sl_corrected.fill(np.nan)
    for count,curr_datetime in enumerate(tgauge_datetime):
        curr_ERA5_ind = (np.abs(ERA5_datetimes - curr_datetime) < np.timedelta64(2,'D')).nonzero()[0]
        if len(curr_ERA5_ind) == 1:
            tgauge_sl_corrected[count] = tgauge_sl[count] + (mon_slp_minus_ocemean[curr_ERA5_ind[0]]/(rho*g))
    
    return tgauge_sl_corrected



def gmsl_sat_read(gmsl_file_path):
    
    with open(gmsl_file_path,'r') as file:
        gmsl_sat_lines = file.readlines()    
    
    date_gmsl_sat = np.empty((0,),dtype='datetime64[ns]')
    gmsl_sat_no_gia = np.array([])
    gmsl_sat_no_gia_smoothed = np.array([])
    gmsl_sat_with_gia = np.array([])
    gmsl_sat_with_gia_smoothed = np.array([])
    in_header_flag = True
    for curr_line in gmsl_sat_lines:
        if in_header_flag == False:
            curr_line_list = curr_line.split("\n")[0].split(" ")
            curr_line_items = []
            for entry in curr_line_list:
                if entry != '':
                    curr_line_items.append(entry)
            curr_date = curr_line_items[2]
            curr_yearstr = curr_date[:4]
            curr_year_length = np.datetime64(str(int(curr_yearstr)+1)+'-01-01','ns')\
                                    - np.datetime64(curr_yearstr+'-01-01','ns')
            date_gmsl_sat = np.append(date_gmsl_sat,\
                                      np.datetime64(curr_yearstr+'-01-01','ns')\
                                      + (float(curr_date[4:])*curr_year_length))
            gmsl_sat_no_gia = np.append(gmsl_sat_no_gia,(1.e-3)*float(curr_line_items[5]))
            gmsl_sat_no_gia_smoothed = np.append(gmsl_sat_no_gia_smoothed,(1.e-3)*float(curr_line_items[7]))
            gmsl_sat_with_gia = np.append(gmsl_sat_with_gia,(1.e-3)*float(curr_line_items[8]))
            gmsl_sat_with_gia_smoothed = np.append(gmsl_sat_with_gia_smoothed,(1.e-3)*float(curr_line_items[10]))
        if 'Header_End' in curr_line:
            in_header_flag = False

    gmsl_sat_no_gia[gmsl_sat_no_gia > 90000] = np.nan
    gmsl_sat_no_gia_smoothed[gmsl_sat_no_gia_smoothed > 90000] = np.nan
    gmsl_sat_with_gia[gmsl_sat_with_gia > 90000] = np.nan
    gmsl_sat_with_gia_smoothed[gmsl_sat_with_gia_smoothed > 90000] = np.nan
    
    return date_gmsl_sat,gmsl_sat_no_gia,gmsl_sat_no_gia_smoothed,gmsl_sat_with_gia,gmsl_sat_with_gia_smoothed


def mask_regionavg_load(file_llc_mask):
    """Read in ECCO grid mask for current region."""
    
    mask_unshaped = np.fromfile(file_llc_mask,dtype='>f4')
    mask_regionavg = np.empty((13,90,90),dtype=np.float32)
    mask_regionavg[:7,:,:] = np.reshape(mask_unshaped[:(7*90*90)],(7,90,90))
    mask_rot = np.reshape(mask_unshaped[(7*90*90):],(180,270))
    for tile_ind in range(7,10):
        mask_regionavg[tile_ind,:,:] = mask_rot[:90,90*(tile_ind-7):90*(tile_ind-6)]
    for tile_ind in range(10,13):
        mask_regionavg[tile_ind,:,:] = mask_rot[90:,90*(tile_ind-10):90*(tile_ind-9)]

    return mask_regionavg


def ecco_mask_in_latlon_grid(file_grid,mask_regionavg,ds_latlon_grid):
    """Indices and area-weighting needed to apply ECCO coast mask in lat-lon gridded data (e.g., altimetry)."""

    ds_ECCO_grid = xr.open_dataset(file_grid)
    
    # indices in ECCO coastal mask
    ECCO_ind_in_mask = (mask_regionavg > .01).nonzero()

    # in masked-region ECCO cell boundaries

    mask_cell_lon_bnds = ds_ECCO_grid.XC_bnds.values[ECCO_ind_in_mask+(slice(None),)]
    # make sure there is no branch cut across cell lon coordinates
    mask_cell_lon_mod_basis = np.nanmin(mask_cell_lon_bnds) - 180
    mask_cell_lon_bnds = ((mask_cell_lon_bnds - mask_cell_lon_mod_basis) % 360)\
                            + mask_cell_lon_mod_basis

    mask_cell_lat_bnds = ds_ECCO_grid.YC_bnds.values[ECCO_ind_in_mask+(slice(None),)]

    mask_cell_lon_outside_bnds = np.array([np.nanmin(mask_cell_lon_bnds),\
                                           np.nanmax(mask_cell_lon_bnds)])
    mask_cell_lat_outside_bnds = np.array([np.nanmin(mask_cell_lat_bnds),\
                                           np.nanmax(mask_cell_lat_bnds)])

    # latlon gridded dataset coordinates and cell boundaries
    llgridded_lon = np.pad(ds_latlon_grid.lon.values,pad_width=(0,1),mode='wrap').astype('float64')
    llgridded_lon = np.hstack((llgridded_lon[0],\
                                llgridded_lon[:-1]\
                                + (((np.diff(llgridded_lon) - (-180)) % 360) - 180)))

    llgridded_lon_bnds = llgridded_lon[:-1] + (np.diff(llgridded_lon)/2)
    llgridded_lon_bnds = np.pad(llgridded_lon_bnds,pad_width=(1,0),mode='wrap')
    llgridded_lon_bnds[0] = llgridded_lon_bnds[1] - (((np.diff(llgridded_lon_bnds[:2])[0] - (-180)) % 360) - 180)
    llgridded_lon = llgridded_lon[:-1]

    llgridded_lat = ds_latlon_grid.lat.values
    llgridded_lat_bnds = llgridded_lat[:-1] + (np.diff(llgridded_lat)/2)

    if np.nanmean(np.diff(llgridded_lon_bnds)) < 0:
        llgridded_lon_bnds = llgridded_lon_bnds[::-1]
    if np.nanmean(np.diff(llgridded_lat_bnds)) < 0:
        llgridded_lat_bnds = llgridded_lat_bnds[::-1]

    llgridded_lat_bnds = np.hstack((np.array([-90]),llgridded_lat_bnds,np.array([90])))

    def llgridded_cell_ind_in_bounds(llgridded_lon_bnds,llgridded_lat_bnds,search_lon_bnds,search_lat_bnds):
        llgridded_inregion_lon_ind = \
            np.logical_or((llgridded_lon_bnds[:-1] - (search_lon_bnds[0] + 1.e-5)) % 360\
                           < ((np.diff(search_lon_bnds) - 2.e-5) % 360),\
                          (llgridded_lon_bnds[1:] - (search_lon_bnds[0] + 1.e-5)) % 360\
                           < ((np.diff(search_lon_bnds) - 2.e-5) % 360)).nonzero()[0]
        llgridded_inregion_lat_ind = np.logical_and(llgridded_lat_bnds[1:] > search_lat_bnds[0] + 1.e-5,\
                                                  llgridded_lat_bnds[:-1] < search_lat_bnds[1] - 1.e-5).nonzero()[0]
        if np.nanmax(np.diff(llgridded_inregion_lon_ind)) > 5:
            gap_ind = np.argmax(np.diff(llgridded_inregion_lon_ind))
            llgridded_inregion_lon_ind = np.hstack((llgridded_inregion_lon_ind[(gap_ind+1):],\
                                                  llgridded_inregion_lon_ind[:(gap_ind+1)]))

        return llgridded_inregion_lon_ind,llgridded_inregion_lat_ind


    llgridded_in_maskregion_lon_ind,llgridded_in_maskregion_lat_ind = \
        llgridded_cell_ind_in_bounds(llgridded_lon_bnds,llgridded_lat_bnds,\
                                   mask_cell_lon_outside_bnds,mask_cell_lat_outside_bnds)

    llgridded_in_maskregion_lon = llgridded_lon[llgridded_in_maskregion_lon_ind]
    llgridded_in_maskregion_lon_bnds = np.asarray(list(llgridded_lon_bnds[llgridded_in_maskregion_lon_ind[0]\
                                                                  :(llgridded_in_maskregion_lon_ind[-1]+2)]))
    llgridded_in_maskregion_lat = llgridded_lat[llgridded_in_maskregion_lat_ind]
    llgridded_in_maskregion_lat_bnds = np.asarray(list(llgridded_lat_bnds[llgridded_in_maskregion_lat_ind[0]\
                                                                  :(llgridded_in_maskregion_lat_ind[-1]+2)]))

    # tuple_ECCO_ind_in_mask = tuple(np.asarray(ECCO_ind_in_mask).transpose())
    area_in_llgridded_cells = np.zeros((len(llgridded_in_maskregion_lat),len(llgridded_in_maskregion_lon)))
    for cell_num in range(len(ECCO_ind_in_mask[0])):
        curr_mask_cell_lon_bnds = np.array([np.nanmin(mask_cell_lon_bnds[cell_num,:]),\
                                            np.nanmax(mask_cell_lon_bnds[cell_num,:])])
        curr_mask_cell_lat_bnds = np.array([np.nanmin(mask_cell_lat_bnds[cell_num,:]),\
                                            np.nanmax(mask_cell_lat_bnds[cell_num,:])])
        llgridded_in_currcell_lon_ind,llgridded_in_currcell_lat_ind = \
            llgridded_cell_ind_in_bounds(llgridded_in_maskregion_lon_bnds,llgridded_in_maskregion_lat_bnds,\
                                       curr_mask_cell_lon_bnds,curr_mask_cell_lat_bnds)
        llgridded_all_in_currcell_lon_ind = llgridded_in_maskregion_lon_ind[llgridded_in_currcell_lon_ind]
        llgridded_all_in_currcell_lat_ind = llgridded_in_maskregion_lat_ind[llgridded_in_currcell_lat_ind]
        llgridded_in_currcell_lon_bnds = np.asarray(list(llgridded_lon_bnds[llgridded_all_in_currcell_lon_ind[0]:\
                                                             (llgridded_all_in_currcell_lon_ind[-1]+2)]))

        curr_lon_mod_basis = llgridded_in_currcell_lon_bnds[0]
        llgridded_in_currcell_lon_bnds = ((llgridded_in_currcell_lon_bnds - curr_lon_mod_basis) % 360)\
                                            + curr_lon_mod_basis
        llgridded_in_currcell_lat_bnds = np.asarray(list(llgridded_lat_bnds[llgridded_all_in_currcell_lat_ind[0]:\
                                                             (llgridded_all_in_currcell_lat_ind[-1]+2)]))

        llgridded_in_currcell_lon_bnds[((llgridded_in_currcell_lon_bnds - curr_mask_cell_lon_bnds[0] - (-180))\
                                       % 360) - 180 < 0] = \
                                        ((curr_mask_cell_lon_bnds[0] - curr_lon_mod_basis) % 360)\
                                        + curr_lon_mod_basis
        llgridded_in_currcell_lon_bnds[((llgridded_in_currcell_lon_bnds - curr_mask_cell_lon_bnds[1] - (-180))\
                                       % 360) - 180 > 0] = \
                                        ((curr_mask_cell_lon_bnds[1] - curr_lon_mod_basis) % 360)\
                                        + curr_lon_mod_basis
        llgridded_in_currcell_lat_bnds[llgridded_in_currcell_lat_bnds < curr_mask_cell_lat_bnds[0]] = \
                                        curr_mask_cell_lat_bnds[0]
        llgridded_in_currcell_lat_bnds[llgridded_in_currcell_lat_bnds > curr_mask_cell_lat_bnds[1]] = \
                                        curr_mask_cell_lat_bnds[1]
        
        curr_area_in_llgridded_cells = (111100**2)*np.reshape(np.cos((np.pi/180)\
                                                                     *llgridded_lat[llgridded_all_in_currcell_lat_ind])\
                                                       *np.diff(llgridded_in_currcell_lat_bnds),(-1,1))\
                                                *np.diff(llgridded_in_currcell_lon_bnds)
        area_in_llgridded_cells[np.reshape(llgridded_in_currcell_lat_ind,(-1,1)),llgridded_in_currcell_lon_ind] += \
                                                        curr_area_in_llgridded_cells

    sum_ssh_nanmask = (~np.isnan(ds_latlon_grid.ssh\
                                   .isel(lat=llgridded_in_maskregion_lat_ind,\
                                         lon=llgridded_in_maskregion_lon_ind))).sum('time').compute()
    area_in_llgridded_cells = np.where(sum_ssh_nanmask > 0.9*ds_latlon_grid.sizes['time'],\
                                      area_in_llgridded_cells,0)
    
    ind_area_dict = {'lat':llgridded_in_maskregion_lat_ind,'lon':llgridded_in_maskregion_lon_ind,\
                     'area':area_in_llgridded_cells}
    
    return ind_area_dict


def pred_endmo_noleap_to_midmo_withleap(ds_curr):
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


def pred_model_lead_sort(curr_da,years_to_predict,n_postinit_leads):
    # sort lead times associated with forcing predictions
    init_time_array = curr_da.init_time.values
    init_time_spacing = np.nanmean(np.diff(init_time_array))
    pred_time_array = curr_da.pred_time.values
    in_pred_range = np.logical_and(pred_time_array >= np.datetime64(str(years_to_predict[0])+'-01-01','ns'),\
                                   pred_time_array < np.datetime64(str(years_to_predict[-1]+1)+'-01-01','ns'))\
                                    .nonzero()
    pred_time_in_range_unique = np.sort(np.unique(pred_time_array[in_pred_range].astype('datetime64[D]')))
    ref_datetime = np.empty((len(pred_time_in_range_unique),),dtype='datetime64[ns]')
    ref_datetime.fill('NaT')
    ref_init_time = np.empty((n_postinit_leads,len(pred_time_in_range_unique)),dtype='datetime64[ns]')
    ref_init_time.fill('NaT')
    ref_tseries = np.empty((n_postinit_leads,len(pred_time_in_range_unique)))
    ref_tseries.fill(np.nan)
    for pred_ind,curr_pred_time in enumerate(pred_time_in_range_unique):
        curr_init_ind,curr_t_ind = \
                            (np.abs(pred_time_array - curr_pred_time) < np.timedelta64(1,'D')).nonzero()
        ref_datetime[pred_ind] = pred_time_array[curr_init_ind[0],curr_t_ind[0]]
        
        curr_postinit_leads = np.empty((len(curr_init_ind),),dtype='timedelta64[ns]')
        curr_postinit_leads.fill('NaT')
        for lead_count,(curr_init,curr_t) in enumerate(zip(curr_init_ind,curr_t_ind)):                
            curr_postinit_leads[lead_count] = curr_pred_time - init_time_array[curr_init]
        min_postinit_leads = np.nanmin(curr_postinit_leads)
        curr_postinit_lead_ind = (np.round((curr_postinit_leads - min_postinit_leads)/init_time_spacing)\
                                    + np.floor(min_postinit_leads/init_time_spacing)).astype('int64')
        for lead_count,curr_init,curr_t in zip(curr_postinit_lead_ind,curr_init_ind,curr_t_ind):
            ref_init_time[lead_count,pred_ind] = init_time_array[curr_init]
            ref_tseries[lead_count,pred_ind] = curr_da.isel(init_time=curr_init,time=curr_t).values

    return ref_tseries,ref_datetime,ref_init_time





def tseries_reference(tseries_opt,years_to_predict,**kwargs):
    
    # # read/compute obs or reconstr time series (if specified by tseries_opt)

    if tseries_opt == 'obs_tgauge':

        file_slp_monthly = kwargs['file_slp_monthly']
        tgauge_obs_filename = kwargs['tgauge_obs_filename']
        lat_pt = kwargs['lat_pt']
        lon_pt = kwargs['lon_pt']
        file_gmsl_sat_era = kwargs['file_gmsl_sat_era']
        file_gmsl_pre_sat = kwargs['file_gmsl_pre_sat']
    
        # # ERA5 monthly mean SLP (for IB correction)
        ds_ERA5_mon_slp = xr.open_mfdataset(file_slp_monthly)        
        
        # retrieve tide gauge observations
        tgauge_SSH_precorr,tgauge_datetime = tgauge_mon_read(tgauge_obs_filename)
    
        # limit to tide gauge months in specified time range
        time_start = np.datetime64(str(years_to_predict[0])+'-01-01','ns')
        time_end = np.datetime64(str(years_to_predict[-1]+1)+'-01-01','ns')
        tgauge_in_time_range = np.logical_and(tgauge_datetime - time_start >= np.timedelta64(0,'ns'),\
                                              tgauge_datetime - time_end < np.timedelta64(0,'ns')).nonzero()[0]
        tgauge_datetime = tgauge_datetime[tgauge_in_time_range]
        tgauge_SSH_precorr = tgauge_SSH_precorr[tgauge_in_time_range]
    
        # apply IB correction
        tgauge_SSH_corrected = tgauge_IB_correction(tgauge_SSH_precorr,tgauge_datetime,lat_pt,lon_pt,ds_ERA5_mon_slp)
        
        
        # # remove global mean SL from tide gauge
        
        gmsl_fields = gmsl_sat_read(file_gmsl_sat_era)
        date_gmsl_sat = gmsl_fields[0]
        gmsl_sat_with_gia = gmsl_fields[3]    
           
        sat_in_time_range = np.logical_and(tgauge_datetime >= np.fmax(time_start,np.datetime64('1992-12-31','ns')),\
                                           tgauge_datetime < time_end).nonzero()[0]

        gmsl_in_sat_range = np.array([])
        for curr_datetime in tgauge_datetime[sat_in_time_range]:
            curr_tbin_start = curr_datetime.astype('datetime64[M]').astype('datetime64[ns]')
            curr_tbin_end = (curr_datetime.astype('datetime64[M]') + np.timedelta64(1,'M')).astype('datetime64[ns]')
            curr_gmsl_sat_ind = np.logical_and(date_gmsl_sat >= curr_tbin_start,\
                                               date_gmsl_sat < curr_tbin_end).nonzero()[0]
            gmsl_in_sat_range = np.append(gmsl_in_sat_range,\
                                          np.nanmean(gmsl_sat_with_gia[curr_gmsl_sat_ind]))
    
        gmsl_in_time_range = gmsl_in_sat_range + 0
    
        # estimate GMSL for times before 1993 (from Frederikse et al. 2020 estimate)
        if time_start < np.datetime64('1992-12-15','ns'):
            # compute seasonal cycle and 1993-2000 trend of GMSL to extrapolate GMSL before 1993
            times_before_sat_era = tgauge_datetime[tgauge_datetime < np.datetime64('1992-12-31','ns')]
            _,gmsl_presat_seas_cyc = seasonal_cycle_harmonics(gmsl_sat_with_gia,\
                                                             date_gmsl_sat,\
                                                             num_harmonics=4,\
                                                             seasonal_output_times=times_before_sat_era,\
                                                             time_axis_num=0)
            gmsl_presat_seas_cyc -= trend_compute_dimvec(gmsl_presat_seas_cyc,\
                                                          times_before_sat_era.astype('float64')/8.64e13,\
                                                          axis=-1,output_trendline=True)[1]
            
            ds_gmsl_pre_sat = xr.open_dataset(file_gmsl_pre_sat)
            # adjust pre-altimetry GMSL estimates by difference with altimetry during 1993-2000
            offset_range_start = np.datetime64('1993-01-01','ns')
            offset_range_end = np.datetime64('2001-01-01','ns')
            gmsl_sat_curr_range_ind = np.logical_and(date_gmsl_sat >= offset_range_start,\
                                                     date_gmsl_sat < offset_range_end).nonzero()[0]
            gmsl_presat_curr_range_ind = np.logical_and(ds_gmsl_pre_sat.time.values >= offset_range_start,\
                                                        ds_gmsl_pre_sat.time.values < offset_range_end).nonzero()[0]
            gmsl_presat_offset = np.nanmean(gmsl_sat_with_gia[gmsl_sat_curr_range_ind])\
                                    - np.nanmean((1.e-3)*ds_gmsl_pre_sat.global_average_sea_level_change\
                                                 [gmsl_presat_curr_range_ind].values)
    
            gmsl_pre_sat_range = np.interp(times_before_sat_era.astype('datetime64[ns]').astype('float64')/8.64e13,\
                                           ds_gmsl_pre_sat.time.values.astype('datetime64[ns]').astype('float64')/8.64e13,\
                                           (1.e-3)*ds_gmsl_pre_sat.global_average_sea_level_change.values)\
                                            + gmsl_presat_offset + gmsl_presat_seas_cyc
            
            gmsl_in_time_range = np.hstack((gmsl_pre_sat_range,gmsl_in_time_range))
        
        tgauge_SSH_rel = tgauge_SSH_corrected - gmsl_in_time_range        
    
        # specify reference time series and time coordinates
        ref_tseries = tgauge_SSH_rel
        ref_datetime = tgauge_datetime


    if tseries_opt == 'obs_altim':

        file_llc_mask = kwargs['file_llc_mask']
        file_obs_alt_gridded = kwargs['file_obs_alt_gridded']
        file_grid = kwargs['file_grid']
        file_gmsl_sat_era = kwargs['file_gmsl_sat_era']

        mask_regionavg = mask_regionavg_load(file_llc_mask)
        
        ds_obs_alt_gridded = xr.open_mfdataset(file_obs_alt_gridded,\
                                    chunks={'time':72})
        
        # compute area-weighted average of SSH in current region
        
        regionavg_alt_ind_area_dict = ecco_mask_in_latlon_grid(file_grid,mask_regionavg,ds_obs_alt_gridded)    
        
        alt_datetime = ds_obs_alt_gridded.time.values
        
        # limit to months in specified time range
        time_start = np.datetime64(str(years_to_predict[0])+'-01-01','ns')
        time_end = np.datetime64(str(years_to_predict[-1]+1)+'-01-01','ns')
        alt_in_time_range = np.logical_and(alt_datetime - time_start >= np.timedelta64(0,'ns'),\
                                           alt_datetime - time_end < np.timedelta64(0,'ns')).nonzero()[0]
        alt_datetime = alt_datetime[alt_in_time_range]

        SSH_alt_near_region = ds_obs_alt_gridded.ssh\
                                .isel(time=alt_in_time_range,\
                                      lat=regionavg_alt_ind_area_dict['lat'],\
                                      lon=regionavg_alt_ind_area_dict['lon']).values
        SSH_alt_mask = (~np.isnan(SSH_alt_near_region))
        SSH_alt_areas = regionavg_alt_ind_area_dict['area']
    
        SSH_alt_regionavg = np.nansum(np.nansum(SSH_alt_mask*\
                                                np.expand_dims(SSH_alt_areas,axis=0)\
                                                *SSH_alt_near_region,axis=-2),axis=-1)\
                                /np.nansum(np.nansum(SSH_alt_mask\
                                                     *np.expand_dims(SSH_alt_areas,axis=0),axis=-2),axis=-1)
        
        # remove GMSL from altimetry time series
        
        gmsl_fields = gmsl_sat_read(file_gmsl_sat_era)
        date_gmsl_sat = gmsl_fields[0]
        gmsl_sat_with_gia = gmsl_fields[3]
        
        gmsl_in_time_range = np.array([])
        for curr_datetime in alt_datetime:
            curr_tbin_start = curr_datetime.astype('datetime64[M]').astype('datetime64[ns]')
            curr_tbin_end = (curr_datetime.astype('datetime64[M]') + np.timedelta64(1,'M')).astype('datetime64[ns]')
            curr_gmsl_sat_ind = np.logical_and(date_gmsl_sat >= curr_tbin_start,\
                                               date_gmsl_sat < curr_tbin_end).nonzero()[0]
            gmsl_in_time_range = np.append(gmsl_in_time_range,\
                                          np.nanmean(gmsl_sat_with_gia[curr_gmsl_sat_ind]))
    
        SSH_alt_regionavg -= gmsl_in_time_range
        
        
        ref_datetime = alt_datetime
        ref_tseries = SSH_alt_regionavg


    if tseries_opt == 'model_seas5_ensmean_monthy':

        ref_dir = kwargs['ref_dir']
        
        n_postinit_leads = 8
        
        ds_curr = xr.open_mfdataset(join(ref_dir,'*contUS.zarr'),\
                                    engine='zarr',\
                                    combine='nested',\
                                    concat_dim='init_time',\
                                    compat='override',\
                                    data_vars='minimal',coords='minimal')

        in_init_time_range = np.unique(\
               np.logical_and(ds_curr.pred_time.values - np.datetime64(str(years_to_predict[0])+'-01-01','ns')\
                                   >= -np.timedelta64(7*max_weeks_lead,'D'),\
                                   ds_curr.pred_time.values - np.datetime64(str(years_to_predict[-1]+1)+'-01-10','ns')\
                                   < np.timedelta64(0,'ns')).nonzero()[0])
        curr_da = ds_curr.zos
        curr_da = curr_da.isel(init_time=in_init_time_range)

        # compute SSH at location
        wet_mask = nanmask_create(curr_da.isel(init_time=0,ens_mem=0,time=0).transpose('lat','lon').values)

        SSH_globmean = ds_curr.zos_globmean

        if 'file_llc_mask' in kwargs.keys():

            file_llc_mask = kwargs['file_llc_mask']

            mask_regionavg = mask_regionavg_load(file_llc_mask)
            mask_regionavg = xr.DataArray(mask_regionavg,dims=['tile','j','i'])

            regionavg_alt_ind_area_dict = ecco_mask_in_latlon_grid(file_grid,mask_regionavg,\
                                                                   curr_da.isel(init_time=0,ens_mem=0,time=0))

            SSH_inregion = (ds_grid.hFacC*ds_grid.rA*mask_regionavg*curr_da).sum(['tile','j','i'])\
                                /((ds_grid.hFacC*ds_grid.rA*mask_regionavg).sum(['tile','j','i']))
            SSH_rel_atplace = (SSH_inregion - SSH_globmean).transpose('init_time','time').compute()
        else:
            # find closest "wet" grid cell to point                       
            closest_lat_ind,closest_lon_ind = np.unravel_index(np.nanargmin(wet_mask\
                            *np.abs(((np.cos((np.pi/180)*lat_pt)*(((curr_da.lon - lon_pt + 180) % 360) - 180))\
                                   + (1j*(curr_da.lat - lat_pt))).transpose('lat','lon').values)).flatten(),\
                                                               wet_mask.shape)
            SSH_atpt = curr_da.isel(lat=closest_lat_ind,lon=closest_lon_ind)
            SSH_rel_atplace = (SSH_atpt - SSH_globmean).transpose('init_time','time').compute()

        ref_tseries,ref_datetime,ref_init_time = \
                        pred_model_lead_sort(SSH_rel_atplace,years_to_predict,n_postinit_leads)
                                    


    if 'model_cesm' in tseries_opt:

        ref_dir = kwargs['ref_dir']
        max_weeks_lead = kwargs['max_weeks_lead']
        place_id = kwargs['place_id']
        file_grid = kwargs['file_grid']
        if 'file_llc_mask' in kwargs.keys():
            file_llc_mask = kwargs['file_llc_mask']

        def preprocess_pred_time(ds):
            ds = ds.assign_coords({'pred_time':ds['time'].copy().expand_dims(dim='init_time',axis=0),\
                                   'pred_time_bound':ds['time_bound'].copy()\
                                                               .expand_dims(dim='init_time',axis=0)})
            del ds['time']
            del ds['time_bound']

            return ds
        
        if tseries_opt == 'model_cesm_dple_ensmean':
            
            n_postinit_leads = 6
            
            file_list = glob.glob(join(ref_dir,'*.nc'))
            file_list_sorted = sorted(file_list)
            
            ds_curr = xr.open_mfdataset(file_list_sorted,\
                                            combine='nested',\
                                            concat_dim='init_time',\
                                            preprocess=preprocess_pred_time,\
                                            data_vars=['SSH'],coords=['pred_time','pred_time_bound'],\
                                            decode_times=False)

            pred_times = pred_endmo_noleap_to_midmo_withleap(ds_curr)
            
            ds_curr = ds_curr.assign_coords({'pred_time':\
                                    (['init_time','time'],pred_times)})
            del ds_curr['pred_time_bound']
            
            init_time = np.empty((len(file_list_sorted),),dtype='datetime64[ns]')
            for file_count,filename in enumerate(file_list_sorted):
                curr_init_yrmo = filename.split('.')[-2][:6]
                curr_init_time = np.datetime64(str(curr_init_yrmo)[0:4]+'-'\
                                   +str(curr_init_yrmo)[4:6]+'-01','ns')
                init_time[file_count] = curr_init_time

            curr_da = 0.01*ds_curr.SSH
            
            
        if tseries_opt == 'model_cesm_hrdp_ensmean':
            
            n_postinit_leads = 3
            
            file_list = glob.glob(join(ref_dir,'*.zarr'))
            file_list_sorted = sorted(file_list)
            
            ds_curr = xr.open_mfdataset(file_list_sorted,\
                                            engine='zarr',\
                                            combine='nested',\
                                            concat_dim='init_time',\
                                            preprocess=preprocess_pred_time,\
                                            data_vars=['SSH'],coords=['pred_time','pred_time_bound'],\
                                            decode_times=False)
            
            pred_times = pred_endmo_noleap_to_midmo_withleap(ds_curr)
            
            ds_curr = ds_curr.assign_coords({'pred_time':\
                                    (['init_time','time'],pred_times)})
            del ds_curr['pred_time_bound']
            
            init_time = np.empty((len(file_list_sorted),),dtype='datetime64[ns]')
            for file_count,filename in enumerate(file_list_sorted):
                curr_init_yrmo = filename.split('.')[-4]
                curr_init_time = np.datetime64(str(curr_init_yrmo)[0:4]+'-'\
                                   +str(curr_init_yrmo)[5:7]+'-01','ns')
                init_time[file_count] = curr_init_time

            curr_da = 0.01*ds_curr.SSH.mean('ens_mem')

        
        curr_da = curr_da.assign_coords({'init_time':\
                                (['init_time'],init_time)})
        
        in_init_time_range = np.unique(\
                 np.logical_and(curr_da.pred_time.values - np.datetime64(str(years_to_predict[0])+'-01-01','ns')\
                                       >= -np.timedelta64(7*max_weeks_lead,'D'),\
                                       curr_da.pred_time.values - np.datetime64(str(years_to_predict[-1]+1)+'-01-10','ns')\
                                       < np.timedelta64(0,'ns')).nonzero()[0])
        curr_da = curr_da.isel(init_time=in_init_time_range)

        # compute SSH at location
        ds_grid = xr.open_dataset(file_grid)
        SSH_globmean = (ds_grid.hFacC.isel(k=0)*ds_grid.rA*curr_da).sum(['tile','j','i'])\
                    /((ds_grid.hFacC.isel(k=0)*ds_grid.rA).sum(['tile','j','i']))
        if 'file_llc_mask' in kwargs.keys():
            mask_regionavg = mask_regionavg_load(file_llc_mask)
            mask_regionavg = xr.DataArray(mask_regionavg,dims=['tile','j','i'])

            SSH_inregion = (ds_grid.hFacC.isel(k=0)*ds_grid.rA*mask_regionavg*curr_da).sum(['tile','j','i'])\
                                /((ds_grid.hFacC.isel(k=0)*ds_grid.rA*mask_regionavg).sum(['tile','j','i']))
            SSH_rel_atplace = (SSH_inregion - SSH_globmean).transpose('init_time','time').compute()
        else:
            # find closest "wet" grid cell to point
            wet_mask = ds_grid.hFacC.isel(k=0).where(ds_grid.hFacC.isel(k=0) > 0.5,np.nan).values
            closest_tile,closest_j,closest_i = np.unravel_index(np.nanargmin(wet_mask\
                            *np.abs((np.cos((np.pi/180)*lat_pt)*(((curr_da.XC.values - lon_pt + 180) % 360) - 180))\
                                   + (1j*(curr_da.YC.values - lat_pt)))).flatten(),curr_da.XC.shape)
            SSH_atpt = curr_da.isel(tile=closest_tile,j=closest_j,i=closest_i)
            SSH_rel_atplace = (SSH_atpt - SSH_globmean).transpose('init_time','time').compute()

        ref_tseries,ref_datetime,ref_init_time = \
                                    pred_model_lead_sort(SSH_rel_atplace,years_to_predict,n_postinit_leads)



    if 'reconstr' in tseries_opt:    
        ds_reconstr = xr.open_mfdataset(reconstr_path,engine='zarr')
        ref_tseries = np.moveaxis(ds_reconstr.reconstr_total.values,0,1).flatten()
        ref_datetime = np.moveaxis(ds_reconstr.pred_time.values,0,1).flatten()
        
        time_start = np.datetime64(str(years_to_predict[0])+'-01-01','ns')
        time_end = np.datetime64(str(years_to_predict[-1]+1)+'-01-01','ns')
        ref_in_time_range = np.logical_and(ref_datetime - time_start >= np.timedelta64(0,'ns'),\
                                           ref_datetime - time_end < np.timedelta64(0,'ns')).nonzero()[0]
        ref_datetime = ref_datetime[ref_in_time_range]
        ref_tseries = ref_tseries[ref_in_time_range]
    
    if 'ref_init_time' in locals().keys():
        return ref_tseries,ref_datetime,ref_init_time
    else:
        return ref_tseries,ref_datetime