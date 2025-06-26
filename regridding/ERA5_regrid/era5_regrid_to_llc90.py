import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
from os.path import join,expanduser
import glob


# source_filepath = join(expanduser('~'),'Downloads','ERA5','weekly')
source_filepath = join('/nobackup','adelman','ERA5','weekly')
source_fileform = 'era5_sflux_weekly_'
source_filepathform = join(source_filepath,source_fileform)
source_files = glob.glob(source_filepathform+'2023.nc')

# output_filepath = join(expanduser('~'),'Downloads','ERA5','weekly','llc90_grid')
output_filepath = join('/nobackup','adelman','ERA5','weekly','llc90_grid')
output_fileform = 'era5_sflux_weekly_llc90grid_'
output_filepathform = join(output_filepath,output_fileform)

# source_mask_file = join(expanduser('~'),'Downloads','ERA5','grid','era5_sst_20190102T00.nc')
source_mask_file = join('/nobackup','adelman','ERA5','grid','era5_sst_20190102T00.nc')

# ECCO_grid_filepath = join(expanduser('~'),'Downloads','ECCO_V4r4_PODAAC','ECCO_L4_GEOMETRY*')
ECCO_grid_filepath = join('/nobackup','adelman','LLC90','grid')
ECCO_grid_file = glob.glob(join(ECCO_grid_filepath,'*.nc'))[0]


# if True, create new indexing/weight arrays to be archived
# if False, read indexing/weight arrays from existing file
create_indexing_weight = False

# regridding/mapping file with indices/weights, either to create or to read from
mapping_file = join('/nobackup','adelman','ERA5','grid','ERA5_p25grid_to_llc90.nc')



# geographic sectors for processing source data
if create_indexing_weight == True:
    sector_lat_bounds = np.hstack((np.arange(-90,-80,1),np.arange(-80,80,2),np.arange(80,91,1)))
else:
    sector_lat_bounds = np.hstack((np.arange(-90,-80,1),np.arange(-80,80,10),np.arange(80,91,1)))
sector_lon_bounds = np.arange(0,370,10)

# default buffer to include (in degrees), this gets modified near the poles
deg_buffer = 0.625



# # functions to find source indices and weights for each ECCO grid cell

def ECCOcell_weight_nearpole(XC_bounds,YC_bounds,ECCO_ind_dict,\
                             fine_X_poleconv,fine_Y_poleconv,fine_array_area,\
                             wet_ECCO,fine_src_mask_array,fine_src_flat_ind_array):
    # convert to pole-centered Cartesian coordinates
    if np.nanmean(YC_bounds) < 0:
        complex_bounds_poleconv = (YC_bounds - (-90))*np.exp(1j*(np.pi/180)*(-XC_bounds))
    else:
        complex_bounds_poleconv = (90 - YC_bounds)*np.exp(1j*(np.pi/180)*XC_bounds)                    
    XC_bounds_poleconv = np.real(complex_bounds_poleconv)
    YC_bounds_poleconv = np.imag(complex_bounds_poleconv)
    # apply successive masks to determine source points/cells in ECCO cell
    YC_diff = np.diff(np.pad(YC_bounds_poleconv,(0,1),'wrap'))
    XC_diff = np.diff(np.pad(XC_bounds_poleconv,(0,1),'wrap'))
    bound_angles = np.arctan2(YC_diff,XC_diff)
    in_ECCO_cell_mask = np.ones((fine_array_area.shape))
    for corner_n,(corner_X,corner_Y) \
      in enumerate(zip(XC_bounds_poleconv,YC_bounds_poleconv)):
        angle_in = bound_angles[[-1,0,1,2]][corner_n] - np.pi
        angle_out = ((bound_angles[corner_n] - (angle_in + 1.e-5)) % (2*np.pi))\
                        + (angle_in + 1.e-5)
        fine_pts_angle = np.arctan2(fine_Y_poleconv - corner_Y,\
                                       fine_X_poleconv - corner_X)
        fine_pts_angle = ((fine_pts_angle - (angle_in + 1.e-5)) % (2*np.pi))\
                        + (angle_in + 1.e-5)
        in_ECCO_cell_mask[fine_pts_angle < angle_out] = np.nan

    # identify source cells in ECCO cell of same type
    src_wetdry_mask = np.where(~np.logical_xor(wet_ECCO >= 0.5,\
                                                  fine_src_mask_array),1,np.nan)
    src_flat_unique = np.unique((src_wetdry_mask*in_ECCO_cell_mask)*fine_src_flat_ind_array)
    apply_wetdry = True
    if np.sum(~np.isnan(src_flat_unique)) < 0.5:
        # if there are no points in ECCO cell with same wet/dry status,
        # ignore the wet/dry mask
        src_flat_unique = np.unique(in_ECCO_cell_mask*fine_src_flat_ind_array)
        apply_wetdry = False
    src_flat_unique = (src_flat_unique[~np.isnan(src_flat_unique)]).astype('int64')
    # create dictionary of cell indices and weightings
    curr_ECCOcell_weighting = {'src_latind':[],\
                               'src_lonind':[],\
                               'src_weight':np.array([])}
    for curr_src_flat in src_flat_unique:
        fine_in_curr_src = (fine_src_flat_ind_array == curr_src_flat)
        curr_ECCOcell_weighting['src_latind'] \
            += [src_lat_ind[int(np.floor(curr_src_flat/len(src_lon_ind)))].astype('int32')]
        curr_ECCOcell_weighting['src_lonind'] \
            += [src_lon_ind[int((curr_src_flat % len(src_lon_ind)))].astype('int32')]
        if apply_wetdry == True:
            curr_ECCOcell_weighting['src_weight'] \
                = np.hstack((curr_ECCOcell_weighting['src_weight'],\
                             np.array([np.nansum(src_wetdry_mask*in_ECCO_cell_mask\
                                                   *fine_in_curr_src*fine_array_area)]).astype('float32')))
        else:
            curr_ECCOcell_weighting['src_weight'] \
                = np.hstack((curr_ECCOcell_weighting['src_weight'],\
                             np.array([np.nansum(in_ECCO_cell_mask\
                                                   *fine_in_curr_src*fine_array_area)]).astype('float32')))
    curr_ECCOcell_weighting['src_weight'] \
                = curr_ECCOcell_weighting['src_weight']/(np.nansum(curr_ECCOcell_weighting['src_weight']))

    return curr_ECCOcell_weighting


def ECCOcell_weight_nearlatlon(XC_bounds,YC_bounds,ECCO_ind_dict,\
                               src_lat_bounds,src_lon_bounds,\
                               src_mask_currbox,wet_ECCO):
    # identify ECCO cell lat/lon bounds
    sorted_curr_YC_bounds = np.sort(YC_bounds)
    ECCOcell_lat_bounds = [np.nanmean(sorted_curr_YC_bounds[:2]),\
                           np.nanmean(sorted_curr_YC_bounds[2:])]
    curr_lonbase = curr_XC_bounds[0] - 10
    sorted_curr_XC_bounds = np.sort(((XC_bounds - curr_lonbase) % 360)\
                                    + curr_lonbase)
    ECCOcell_lon_bounds = [np.nanmean(sorted_curr_XC_bounds[:2]),\
                           np.nanmean(sorted_curr_XC_bounds[2:])]

    # identify source cell lat/lon bounds within current ECCO cell
    src_lat_bounds_inECCO_ind = np.logical_and(src_lat_bounds > ECCOcell_lat_bounds[0],\
                                               src_lat_bounds < ECCOcell_lat_bounds[1]).nonzero()[0]
    src_lat_bounds_inECCO_ind = src_lat_bounds_inECCO_ind\
                                [np.argsort(src_lat_bounds[src_lat_bounds_inECCO_ind])]
    src_lon_bounds_inECCO_ind = np.logical_and(((src_lon_bounds - curr_lonbase) % 360)\
                                                   + curr_lonbase > ECCOcell_lon_bounds[0],\
                                                   ((src_lon_bounds - curr_lonbase) % 360)\
                                                   + curr_lonbase < ECCOcell_lon_bounds[1]).nonzero()[0]
    src_lon_bounds_inECCO_ind = src_lon_bounds_inECCO_ind\
                                [np.argsort(src_lon_bounds[src_lon_bounds_inECCO_ind])]
    src_lat_bounds_inECCO = src_lat_bounds[src_lat_bounds_inECCO_ind]
    src_lon_bounds_inECCO = src_lon_bounds[src_lon_bounds_inECCO_ind]
    src_lat_inECCO_ind = np.hstack((np.array([np.nanmin(src_lat_bounds_inECCO_ind)]),\
                                           src_lat_bounds_inECCO_ind + 1))
    if np.nansum(np.diff(src_lat_bounds)) < 0:
        src_lat_inECCO_ind = np.arange(np.nanmax(src_lat_bounds_inECCO_ind)+1,\
                                       np.nanmin(src_lat_bounds_inECCO_ind)-1,\
                                       -1)
    else:
        src_lat_inECCO_ind = np.arange(np.nanmin(src_lat_bounds_inECCO_ind),\
                                       np.nanmax(src_lat_bounds_inECCO_ind)+2,\
                                       1)
    if np.nansum(np.diff(src_lon_bounds)) < 0:
        src_lon_inECCO_ind = np.arange(np.nanmax(src_lon_bounds_inECCO_ind)+1,\
                                       np.nanmin(src_lon_bounds_inECCO_ind)-1,\
                                       -1)
    else:
        src_lon_inECCO_ind = np.arange(np.nanmin(src_lon_bounds_inECCO_ind),\
                                       np.nanmax(src_lon_bounds_inECCO_ind)+2,\
                                       1)
    src_lon_inECCO_ind = np.hstack((np.array([np.nanmin(src_lon_bounds_inECCO_ind)]),\
                                           src_lon_bounds_inECCO_ind + 1))
    src_lat_bounds_inECCO = np.hstack((np.array([ECCOcell_lat_bounds[0]]),\
                                       src_lat_bounds_inECCO,\
                                       np.array([ECCOcell_lat_bounds[1]])))
    src_lon_bounds_inECCO = np.hstack((np.array([ECCOcell_lon_bounds[0]]),\
                                       ((src_lon_bounds_inECCO - curr_lonbase) % 360) + curr_lonbase,\
                                       np.array([ECCOcell_lon_bounds[1]])))

    # identify source indices (within current region) for current ECCO cell
    src_array_latind_inECCO = np.tile(np.reshape(src_lat_inECCO_ind,(-1,1)),\
                                      (1,len(src_lon_inECCO_ind)))
    src_array_lonind_inECCO = np.tile(src_lon_inECCO_ind,(len(src_lat_inECCO_ind),1))
    src_array_flatind_inECCO = src_array_lonind_inECCO \
                                + (src_array_latind_inECCO*len(src_lon_ind))
    # source wet/dry masks within ECCO cell
    # if ECCO cell is wet (dry) then only wet (dry) source cells will be included
    src_array_wet_inECCO = src_mask_currbox.flatten()[src_array_flatind_inECCO]
    src_array_wetdry_inECCO = ~np.logical_xor(wet_ECCO >= 0.5,src_array_wet_inECCO)
    # source areas within ECCO cell
    src_array_area = (111100**2)*np.cos((np.pi/180)*np.reshape(\
                                              ds_src.latitude.isel(latitude=src_lat_ind)\
                                              .values[src_lat_inECCO_ind],(-1,1)))\
                                *np.diff(src_lon_bounds_inECCO)\
                                *np.reshape(np.diff(src_lat_bounds_inECCO),(-1,1))

    # create dictionary of cell indices (within current region) and weightings
    curr_ECCOcell_weighting = {'src_latind':[],\
                               'src_lonind':[],\
                               'src_weight':np.array([])}
    same_wetdry_ind = (src_array_wetdry_inECCO.flatten() == True).nonzero()[0]
    src_flat_unique = src_array_flatind_inECCO.flatten()[same_wetdry_ind]
    if len(src_flat_unique) == 0:
        # if there are no points in ECCO cell with same wet/dry status,
        # ignore the wet/dry mask
        src_flat_unique = src_array_flatind_inECCO.flatten()
        src_flat_inECCO_j = (np.floor(np.arange(len(src_flat_unique))\
                                      /len(src_lon_inECCO_ind))).astype('int64')
        src_flat_inECCO_i = np.arange(len(src_flat_unique)) % len(src_lon_inECCO_ind)
    else:
        src_flat_inECCO_j = (np.floor(same_wetdry_ind/len(src_lon_inECCO_ind))).astype('int64')
        src_flat_inECCO_i = same_wetdry_ind % len(src_lon_inECCO_ind)
    for curr_src_flat,curr_src_inECCO_j,curr_src_inECCO_i\
      in zip(src_flat_unique,src_flat_inECCO_j,src_flat_inECCO_i):
        curr_ECCOcell_weighting['src_latind'] \
            += [src_lat_ind[int(np.floor(curr_src_flat/len(src_lon_ind)))].astype('int32')]
        curr_ECCOcell_weighting['src_lonind'] \
            += [src_lon_ind[int(curr_src_flat % len(src_lon_ind))].astype('int32')]
        curr_ECCOcell_weighting['src_weight'] \
            = np.hstack((curr_ECCOcell_weighting['src_weight'],\
                         src_array_area[curr_src_inECCO_j,curr_src_inECCO_i].astype('float32')))

    curr_ECCOcell_weighting['src_weight'] \
                = curr_ECCOcell_weighting['src_weight']/\
                    (np.nansum(curr_ECCOcell_weighting['src_weight']))
    
    return curr_ECCOcell_weighting





# load grid and mask files

ds_ECCO_grid = xr.open_mfdataset(ECCO_grid_file)
ds_src_mask = xr.open_dataset(source_mask_file)

src_mask = ~np.isnan(ds_src_mask.sst.values.squeeze())



if create_indexing_weight == True:
    # pre-allocate indexing/weight nested lists
    # to be converted later to NumPy arrays and then a dataset
    for tile_ind in range(ds_ECCO_grid.sizes['tile']):
        tile_str = str(tile_ind)
        exec('ECCO_i_indices_'+tile_str+' = [np.array([]).astype("int32")]*ds_ECCO_grid.sizes["j"]')
        exec('src_latind_'+tile_str+' = [np.array([]).astype("int32")]*ds_ECCO_grid.sizes["j"]')
        exec('src_lonind_'+tile_str+' = [np.array([]).astype("int32")]*ds_ECCO_grid.sizes["j"]')
        exec('src_num_inECCO_'+tile_str+' = [np.array([]).astype("int64")]*ds_ECCO_grid.sizes["j"]')
        exec('src_weight_'+tile_str+' = [np.array([]).astype("float64")]*ds_ECCO_grid.sizes["j"]')
    i_and_src_curr_len = np.zeros((ds_ECCO_grid.sizes['tile'],ds_ECCO_grid.sizes['j'])).astype('int64')
else:
    ds_weights = xr.open_dataset(mapping_file)
    for tile_ind in range(ds_ECCO_grid.sizes['tile']):
        tile_str = str(tile_ind)
        for j_ind in range(ds_ECCO_grid.sizes['j']):
            j_str = str(j_ind)
            exec('ECCO_i_indices_list_'+tile_str+'_'+j_str+' = [()]*ds_ECCO_grid.sizes["i"]')
            curr_nparray = ds_weights.ECCO_i.isel(tile=tile_ind,j=j_ind).values
            good_ind = (np.abs(curr_nparray.astype('int64')) < 1.e6).nonzero()[0]
            for curr_i_and_src,curr_i in zip(good_ind,curr_nparray[good_ind]):
                exec('ECCO_i_indices_list_'+tile_str+'_'+j_str+'[curr_i] += (curr_i_and_src,)')


# process source files
source_files = glob.glob(source_filepathform+'2023.nc')
for file_count,source_file in enumerate(source_files):
    source_file_end = source_file[len(source_filepathform):]
    ds_src = xr.open_mfdataset(source_file,chunks={'latitude':181,'longitude':360})

    # create new dataset corresponding to source file, with llc grid
    ds_new = xr.Dataset(coords=ds_src.coords,\
                        attrs=ds_src.attrs,\
                        )
    ds_new = ds_new.drop_dims(['latitude','longitude'])

    ECCO_dims_to_copy = ['i','i_g','j','j_g','tile']
    ECCO_coords = ds_ECCO_grid.coords
    for curr_coord in list(ECCO_coords):
        for curr_dim in ds_ECCO_grid[curr_coord].dims:
            if curr_dim in ECCO_dims_to_copy:
                ds_new = ds_new.assign_coords({curr_coord:ds_ECCO_grid[curr_coord]})
                break

    data_var_names = list(ds_src.keys())
    new_datavars = dict()
    for varname in data_var_names:
        data_sizes = ds_src[varname].sizes
        dim_list = []
        dimsize_tuple = tuple()
        for curr_dim in data_sizes:
            if curr_dim == 'latitude':
                dim_list.extend(['tile','j'])
                dimsize_tuple += (ds_ECCO_grid.sizes['tile'],ds_ECCO_grid.sizes['j'])
            elif curr_dim == 'longitude':
                dim_list.append('i')
                dimsize_tuple += (ds_ECCO_grid.sizes['i'],)
            else:
                dim_list.append(curr_dim)
                dimsize_tuple += (data_sizes[curr_dim],)
        curr_array = np.empty(dimsize_tuple).astype('float32')
        curr_array.fill(np.nan)
        new_datavars = {**new_datavars,**{varname:[dim_list,\
                                                   curr_array,\
                                                   ds_src[varname].attrs]}}
    

    ## Note: regridding is handled differently near the poles
    ## Away from the poles, the ECCO grid is essentially a lat/lon grid and is assumed to be so.
    ## South of ~70 S and north of ~60 N, the remapping is done by defining "fine-scale" arrays.

    for curr_lat_bounds in zip(sector_lat_bounds[:-1],sector_lat_bounds[1:]):
        if np.nanmax(np.abs(curr_lat_bounds)) > 85:
            lat_deg_buffer = 0.375
        else:
            lat_deg_buffer = deg_buffer
        src_lat_ind = np.logical_and(ds_src.latitude.values - curr_lat_bounds[0] >= -lat_deg_buffer,\
                                      ds_src.latitude.values - curr_lat_bounds[1] <= lat_deg_buffer).nonzero()[0]
        src_lat_bounds = ds_src.latitude[src_lat_ind[:-1]].values\
                            + (np.diff(ds_src.latitude[src_lat_ind].values)/2)
        if ((curr_lat_bounds[0] < -71) or (curr_lat_bounds[1] > 61)):
            # start defining "fine-scale" array
            # to identify ECCO cell bounds in the source grid
            fine_array_spacing = 1/32
            fine_array_lat = np.reshape(np.arange(np.fmax(curr_lat_bounds[0] - lat_deg_buffer,-90)\
                                                  + (fine_array_spacing/2),np.fmin(curr_lat_bounds[1],90),\
                                                   fine_array_spacing),(-1,1))
            
            fine_array_src_lat_ind = np.empty((fine_array_lat.shape[0],1)).astype('int32')
            fine_array_src_lat_ind.fill(-1000000000)
            fine_array_src_lat_ind[fine_array_lat < np.nanmin(src_lat_bounds)]\
                                    = np.argmin(ds_src.latitude[src_lat_ind].values)
            for curr_count,curr_srclat_bounds in enumerate(zip(src_lat_bounds[:-1],src_lat_bounds[1:])):
                curr_lat_ind = np.logical_and(fine_array_lat >= np.nanmin(curr_srclat_bounds),\
                                              fine_array_lat < np.nanmax(curr_srclat_bounds)).nonzero()[0]
                fine_array_src_lat_ind[curr_lat_ind] = curr_count+1
            fine_array_src_lat_ind[fine_array_lat >= np.nanmax(src_lat_bounds)]\
                                    = np.argmax(ds_src.latitude[src_lat_ind].values)

        if np.nanmax(np.abs(curr_lat_bounds)) > 88:
            curr_sector_lon_bounds = np.array([-0.125,359.875])
        elif np.nanmax(np.abs(curr_lat_bounds)) > 86:
            curr_sector_lon_bounds = np.arange(0,450,90)
        elif np.nanmax(np.abs(curr_lat_bounds)) > 81:
            curr_sector_lon_bounds = np.arange(0,390,30)
        else:
            curr_sector_lon_bounds = sector_lon_bounds
        for curr_lon_bounds in zip(curr_sector_lon_bounds[:-1],curr_sector_lon_bounds[1:]):
            if np.nanmax(np.abs(curr_lat_bounds)) > 88:
                lon_deg_buffer = 0
            elif np.nanmax(np.abs(curr_lat_bounds)) > 86:
                lon_deg_buffer = 30*deg_buffer
            elif np.nanmax(np.abs(curr_lat_bounds)) > 81:
                lon_deg_buffer = 10*deg_buffer
            elif np.nanmax(np.abs(curr_lat_bounds)) > 71:
                lon_deg_buffer = 5*deg_buffer
            elif np.nanmax(np.abs(curr_lat_bounds)) > 61:
                lon_deg_buffer = 3*deg_buffer
            else:
                lon_deg_buffer = deg_buffer
            src_lon_ind = ((ds_src.longitude.values - (curr_lon_bounds[0] - lon_deg_buffer) - 1.e-5) % 360 \
                            <= ((curr_lon_bounds[1] + lon_deg_buffer)\
                                - (curr_lon_bounds[0] - lon_deg_buffer) - 1.e-5) % 360).nonzero()[0]
            if np.nanmax(np.abs(np.diff(src_lon_ind))) > 5:
                gap_ind = np.argmax(np.abs(np.diff(src_lon_ind)))
                src_lon_ind = np.hstack((src_lon_ind[(gap_ind+1):],src_lon_ind[:(gap_ind+1)]))
            curr_diff_lon = ((np.diff(ds_src.longitude[src_lon_ind].values) + 180) % 360) - 180
            if np.nansum(curr_diff_lon) < 0:
                lonbase = ds_src.longitude[src_lon_ind[-1]].values - 1.e-5
            else:
                lonbase = ds_src.longitude[src_lon_ind[0]].values - 1.e-5
            src_lon_bounds = ds_src.longitude.values[src_lon_ind[0]]\
                                + ((((np.diff(ds_src.longitude[src_lon_ind[0:2]].values) + 180) % 360) - 180)/2)\
                                + np.hstack((np.array([0]),\
                                             np.cumsum(((((ds_src.longitude[src_lon_ind[2:]].values\
                                                - ds_src.longitude[src_lon_ind[:-2]].values) + 180)\
                                                        % 360) - 180)/2)))
            src_lon_bounds = ((src_lon_bounds - lonbase) % 360) + lonbase

            # load source dataset and wet/dry mask for current region
            ds_src_currbox = ds_src.isel(latitude=src_lat_ind,longitude=src_lon_ind).compute()
            src_mask_currbox = src_mask[np.reshape(src_lat_ind,(-1,1)),src_lon_ind]

            # identify ECCO cells in current region
            ECCO_cells_inbox = np.logical_and((ds_ECCO_grid.XC.values - curr_lon_bounds[0] - 1.e-5) % 360 \
                                              <= (curr_lon_bounds[1] - curr_lon_bounds[0] - 1.e-5) % 360,\
                                              np.logical_and(ds_ECCO_grid.YC.values >= curr_lat_bounds[0],\
                                                             ds_ECCO_grid.YC.values < curr_lat_bounds[1]))\
                                                .nonzero()

            if (create_indexing_weight == True) and ((curr_lat_bounds[0] < -71) or (curr_lat_bounds[1] > 61)):
                # continue defining the "fine-scale" arrays
                fine_array_lon_start = lonbase + (((curr_lon_bounds[0] - lonbase + 180) % 360) - 180)\
                                        - lon_deg_buffer
                fine_array_lon = np.arange(fine_array_lon_start + (fine_array_spacing/2),\
                                           fine_array_lon_start + np.diff(curr_lon_bounds) + (2*lon_deg_buffer),\
                                           fine_array_spacing)
                fine_array_area = np.tile((111100**2)*np.cos((np.pi/180)*fine_array_lat)*(fine_array_spacing**2),\
                                          (1,len(fine_array_lon)))

                fine_array_src_lon_ind = np.empty((fine_array_lon.shape[-1],)).astype('int32')
                fine_array_src_lon_ind.fill(-1000000000)
                fine_array_src_lon_ind[fine_array_lon < np.nanmin(src_lon_bounds)]\
                                        = np.argmin(((ds_src.longitude.values[src_lon_ind] - lonbase) % 360)\
                                                    + lonbase)
                for curr_count,curr_srclon_bounds in enumerate(zip(src_lon_bounds[:-1],src_lon_bounds[1:])):
                    curr_lon_ind = np.logical_and(fine_array_lon >= np.nanmin(curr_srclon_bounds),\
                                                  fine_array_lon < np.nanmax(curr_srclon_bounds)).nonzero()[0]
                    fine_array_src_lon_ind[curr_lon_ind] = curr_count+1
                fine_array_src_lon_ind[fine_array_lon >= np.nanmax(src_lon_bounds)]\
                                        = np.argmax(((ds_src.longitude.values[src_lon_ind] - lonbase) % 360)\
                                                    + lonbase)

                fine_src_lat_ind_array = np.tile(fine_array_src_lat_ind,(1,len(fine_array_src_lon_ind)))
                fine_src_lon_ind_array = np.tile(fine_array_src_lon_ind,(len(fine_array_src_lat_ind),1))
                fine_src_flat_ind_array = fine_src_lon_ind_array\
                                            + (fine_src_lat_ind_array*len(src_lon_ind))
                fine_src_mask_array = src_mask_currbox.flatten()[fine_src_flat_ind_array]

                # convert to pole-centered Cartesian coordinates
                if np.nanmean(curr_lat_bounds) < 0:
                    fine_poleconv = (fine_array_lat - (-90))*np.exp(1j*(np.pi/180)*(-fine_array_lon))
                else:
                    fine_poleconv = (90 - fine_array_lat)*np.exp(1j*(np.pi/180)*fine_array_lon)
                fine_X_poleconv = np.real(fine_poleconv)
                fine_Y_poleconv = np.imag(fine_poleconv)

            for tile_ind,j_ind,i_ind in np.moveaxis(np.asarray(ECCO_cells_inbox),0,1):
                ECCO_ind_dict = {'tile':tile_ind,'j':j_ind,'i':i_ind}
                if create_indexing_weight == True:
                    wet_ECCO = ds_ECCO_grid.hFacC.isel(**{'k':0},**ECCO_ind_dict).values
                    curr_XC_bounds = ds_ECCO_grid.XC_bnds.isel(ECCO_ind_dict).values
                    curr_YC_bounds = ds_ECCO_grid.YC_bnds.isel(ECCO_ind_dict).values
                    if ((curr_lat_bounds[0] < -71) or (curr_lat_bounds[1] > 61)):
                        curr_ECCOcell_weighting = ECCOcell_weight_nearpole(curr_XC_bounds,curr_YC_bounds,\
                                                                           ECCO_ind_dict,\
                                                                           fine_X_poleconv,fine_Y_poleconv,\
                                                                           fine_array_area,\
                                                                           wet_ECCO,fine_src_mask_array,\
                                                                           fine_src_flat_ind_array)

                    else:
                        curr_ECCOcell_weighting = ECCOcell_weight_nearlatlon(curr_XC_bounds,curr_YC_bounds,\
                                                                             ECCO_ind_dict,\
                                                                             src_lat_bounds,src_lon_bounds,\
                                                                             src_mask_currbox,wet_ECCO)

                    # add current ECCO cell indices/weighting to cumulative arrays
                    curr_len_to_add = len(curr_ECCOcell_weighting['src_weight'])
                    tile_str = str(tile_ind)
                    exec('ECCO_i_indices_'+tile_str+'[j_ind] = np.hstack((ECCO_i_indices_'+tile_str+'[j_ind],'\
                                                                      + 'i_ind*(np.ones((curr_len_to_add,))'\
                                                                      + '.astype("int32"))))')
                    exec('src_latind_'+tile_str+'[j_ind] = np.hstack((src_latind_'+tile_str+'[j_ind],'\
                                                             + 'np.asarray(curr_ECCOcell_weighting["src_latind"])'\
                                                             + '.astype("int32")))')
                    exec('src_lonind_'+tile_str+'[j_ind] = np.hstack((src_lonind_'+tile_str+'[j_ind],'\
                                                             + 'np.asarray(curr_ECCOcell_weighting["src_lonind"])'\
                                                             + '.astype("int32")))')
                    exec('src_num_inECCO_'+tile_str+'[j_ind] = np.hstack((src_num_inECCO_'+tile_str+'[j_ind],'\
                                                                  + 'np.arange(0,curr_len_to_add)'\
                                                                  + '.astype("int64")))')
                    exec('src_weight_'+tile_str+'[j_ind] = np.hstack((src_weight_'+tile_str+'[j_ind],'\
                                                                  + 'curr_ECCOcell_weighting["src_weight"]'\
                                                                  + '.astype("float32")))')
                    i_and_src_curr_len[tile_ind,j_ind] += curr_len_to_add

                else:
                    # load indexing/weight information from existing dataset
                    curr_ECCOcell_weighting = dict()
                    tile_str = str(tile_ind)
                    j_str = str(j_ind)
                    curr_i_and_src_ind = eval('list(ECCO_i_indices_list_'+tile_str+'_'+j_str+'[i_ind])')
                    if len(curr_i_and_src_ind) == 0:
                        continue
                    src_ind_dict = {'tile':tile_ind,'j':j_ind,'i_and_src':curr_i_and_src_ind}
                    ds_weights_at_curr_ECCO_i = ds_weights.isel(src_ind_dict)
                    curr_ECCOcell_weighting['src_latind'] = ds_weights_at_curr_ECCO_i.src_latind.values
                    curr_ECCOcell_weighting['src_lonind'] = ds_weights_at_curr_ECCO_i.src_lonind.values
                    curr_ECCOcell_weighting['src_weight'] = ds_weights_at_curr_ECCO_i.src_weight.values

                # find indices in regional sub-arrays
                curr_src_sublatind = np.array([]).astype('int32')
                curr_src_sublonind = np.array([]).astype('int32')
                for curr_srclatind,curr_srclonind in zip(curr_ECCOcell_weighting['src_latind'],\
                                                         curr_ECCOcell_weighting['src_lonind']):
                    d_ind_from_start = curr_srclatind - src_lat_ind[0]
                    if (d_ind_from_start <= 0) and \
                      (src_lat_ind[-d_ind_from_start] == curr_srclatind):
                        curr_src_sublatind = np.hstack((curr_src_sublatind,np.array([-d_ind_from_start])))
                    elif (d_ind_from_start >= 0) and \
                      (src_lat_ind[d_ind_from_start] == curr_srclatind):
                        curr_src_sublatind = np.hstack((curr_src_sublatind,np.array([d_ind_from_start])))
                    else:
                        curr_ind = (src_lat_ind == curr_srclatind).nonzero()[0]
                        curr_src_sublatind = np.hstack((curr_src_sublatind,np.array([curr_ind])))
                    d_ind_from_start = curr_srclonind - src_lon_ind[0]
                    if (np.abs(d_ind_from_start) >= len(src_lon_ind)):
                        if d_ind_from_start <= 0:
                            d_ind_from_start += ds_src.sizes['longitude']
                        else:
                            d_ind_from_start += -(ds_src.sizes['longitude'])
                    if (d_ind_from_start >= 0) and \
                      (src_lon_ind[d_ind_from_start] == curr_srclonind):
                        curr_src_sublonind = np.hstack((curr_src_sublonind,np.array([d_ind_from_start])))
                    elif (d_ind_from_start <= 0) and \
                      (src_lon_ind[-d_ind_from_start] == curr_srclonind):
                        curr_src_sublonind = np.hstack((curr_src_sublonind,np.array([-d_ind_from_start])))
                    else:
                        curr_ind = (src_lon_ind == curr_srclonind).nonzero()[0]
                        curr_src_sublonind = np.hstack((curr_src_sublonind,np.array([curr_ind])))
                curr_weights = curr_ECCOcell_weighting['src_weight']

                # compute regridding of data variables
                for varname in data_var_names:
                    curr_var_inECCO = ds_src_currbox[varname]
                    dict_sizes = dict(curr_var_inECCO.sizes)
                    del dict_sizes['latitude']
                    del dict_sizes['longitude']

                    curr_var_inECCO_vals = curr_var_inECCO.values
                    curr_var_value = np.zeros(tuple(dict_sizes.values()))
                    for curr_sublatind,curr_sublonind,curr_weight \
                      in zip(curr_src_sublatind,curr_src_sublonind,\
                             curr_weights):
                        curr_var_value += (curr_weight*curr_var_inECCO_vals[...,curr_sublatind,curr_sublonind])
                    new_datavars[varname][1][...,tile_ind,j_ind,i_ind] = curr_var_value



    if create_indexing_weight == True:
        # convert nested lists into NumPy arrays (and JSON-ready lists)
        max_i_src_curr_len = np.nanmax(i_and_src_curr_len)
        # ECCO_i_indices_list = [[]]*ds_ECCO_grid.sizes['tile']
        # src_latind_list = [[]]*ds_ECCO_grid.sizes['tile']
        # src_lonind_list = [[]]*ds_ECCO_grid.sizes['tile']
        # src_num_inECCO_list = [[]]*ds_ECCO_grid.sizes['tile']
        # src_weight_list = [[]]*ds_ECCO_grid.sizes['tile']

        ECCO_i_indices_array = np.empty((ds_ECCO_grid.sizes['tile'],ds_ECCO_grid.sizes['j'],\
                                   max_i_src_curr_len)).astype('int32')
        ECCO_i_indices_array.fill(-1000000000)
        src_latind_array = np.empty((ds_ECCO_grid.sizes['tile'],ds_ECCO_grid.sizes['j'],\
                               max_i_src_curr_len)).astype('int32')
        src_latind_array.fill(-1000000000)
        src_lonind_array = np.empty((ds_ECCO_grid.sizes['tile'],ds_ECCO_grid.sizes['j'],\
                               max_i_src_curr_len)).astype('int32')
        src_lonind_array.fill(-1000000000)
        src_num_inECCO_array = np.empty((ds_ECCO_grid.sizes['tile'],ds_ECCO_grid.sizes['j'],\
                                   max_i_src_curr_len)).astype('int64')
        src_num_inECCO_array.fill(-1000000000)
        src_weight_array = np.empty((ds_ECCO_grid.sizes['tile'],ds_ECCO_grid.sizes['j'],\
                               max_i_src_curr_len)).astype('float32')
        src_weight_array.fill(np.nan)
        for tile_ind in range(ECCO_i_indices_array.shape[0]):
            tile_str = str(tile_ind)
            for j_ind in range(ECCO_i_indices_array.shape[1]):
                curr_len = i_and_src_curr_len[tile_ind,j_ind]
                ECCO_i_indices_array[tile_ind,j_ind,:curr_len] = eval('ECCO_i_indices_'+tile_str+'[j_ind]')
                src_latind_array[tile_ind,j_ind,:curr_len] = eval('src_latind_'+tile_str+'[j_ind]')
                src_lonind_array[tile_ind,j_ind,:curr_len] = eval('src_lonind_'+tile_str+'[j_ind]')
                src_num_inECCO_array[tile_ind,j_ind,:curr_len] = eval('src_num_inECCO_'+tile_str+'[j_ind]')
                src_weight_array[tile_ind,j_ind,:curr_len] = eval('src_weight_'+tile_str+'[j_ind]')

        ds_weights = xr.Dataset(\
                               data_vars={
                                         'src_weight':(['tile','j','i_and_src'],\
                                                       src_weight_array,\
                                                       {'long_name':'weight of index in source file(s)'}),\
                                         },\
                                coords={\
                                       'ECCO_tile':(['tile'],np.arange(ds_ECCO_grid.sizes['tile']),\
                                                    {'long_name':'ECCO cell tile index'}),\
                                       'ECCO_j':(['j'],np.arange(ds_ECCO_grid.sizes['j']),\
                                                 {'long_name':'ECCO cell j index'}),\
                                       'ECCO_i':(['tile','j','i_and_src'],\
                                                ECCO_i_indices_array,\
                                                {'long_name':'ECCO cell i index'}),\
                                       'src_latind':(['tile','j','i_and_src'],\
                                                       src_latind_array,\
                                                       {'long_name':'latitude index in source file(s)'}),\
                                       'src_lonind':(['tile','j','i_and_src'],\
                                                       src_lonind_array,\
                                                       {'long_name':'longitude index in source file(s)'}),\

                                       'src_num_inECCO':(['tile','j','i_and_src'],\
                                                         src_num_inECCO_array,\
                                                         {'long_name':'number of src index in ECCO cell'}),\
                                       },\
                                attrs={'description':'Information for mapping ERA5 source data to ECCO llc90 grid'},\
                               )

        ds_weights.to_netcdf(path=mapping_file,format="NETCDF4")
        
        create_indexing_weight = False          
        ds_weights = xr.open_dataset(mapping_file)
        for tile_ind in range(ds_ECCO_grid.sizes['tile']):
            tile_str = str(tile_ind)
            for j_ind in range(ds_ECCO_grid.sizes['j']):
                j_str = str(j_ind)
                exec('ECCO_i_indices_list_'+tile_str+'_'+j_str+' = [()]*ds_ECCO_grid.sizes["i"]')
                curr_nparray = ds_weights.ECCO_i.isel(tile=tile_ind,j=j_ind).values
                good_ind = (np.abs(curr_nparray.astype('int64')) < 1.e6).nonzero()[0]
                for curr_i_and_src,curr_i in zip(good_ind,curr_nparray[good_ind]):
                    exec('ECCO_i_indices_list_'+tile_str+'_'+j_str+'[curr_i] += (curr_i_and_src,)')



    # write regridded fields to new dataset
    for varname in data_var_names:
        ds_new[varname] = tuple(new_datavars[varname])

    del new_datavars

    # write regridded dataset to output file
    output_file = output_filepathform+source_file_end
    ds_new.to_netcdf(path=output_file,format="NETCDF4")
    print('Created file '+output_file)
        
    del ds_new
