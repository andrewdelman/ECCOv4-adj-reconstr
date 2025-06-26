"""Functions to create dictionaries with nested geometries of merged ECCO cells."""

import numpy as np


def nested_geom_attile(tile,indices,curr_n_merge,cell_indices,array_indexing,incell_order,\
        XC_cell,YC_cell,XG_cell,YG_cell,rA_cell,\
        XC,YC,XG,YG,rA,\
        already_covered):
    curr_j = np.sort(indices['j'])
    curr_i = np.sort(indices['i'])
    start_j = np.arange(curr_j[0],curr_j[-1]+1,curr_n_merge)
    for s_j in start_j:
        incell_j = np.arange(s_j,np.fmin(s_j+curr_n_merge,curr_j[-1]+1))
        start_i = np.arange(curr_i[0],curr_i[-1]+1,curr_n_merge)
        for s_i in start_i:
            incell_i = np.arange(s_i,np.fmin(s_i+curr_n_merge,curr_i[-1]+1))
            n_el = len(incell_j)*len(incell_i)
            
            # create indexing vectors for current cell
            tile_cell_vec = tile*(np.ones((n_el,))).astype('i8')
            j_cell_vec = np.tile(np.reshape(incell_j,(-1,1)),(1,len(incell_i))).flatten()
            i_cell_vec = np.tile(incell_i,(len(incell_j),))
            cell_ind_tuple = (tile_cell_vec,j_cell_vec,i_cell_vec)
            
            # avoid repeating grid cells already covered in a previous scheme
            if np.sum(already_covered[cell_ind_tuple]) == n_el:
                continue
            elif np.sum(already_covered[cell_ind_tuple]) > 0:
                notyet_covered_ind = (already_covered[cell_ind_tuple] < .9999).nonzero()[0]
                cell_ind_tuple = tuple(np.asarray(cell_ind_tuple)[:,notyet_covered_ind])
            
            dict_cell_indices = {}    
            dict_cell_indices['tile'] = cell_ind_tuple[0]
            dict_cell_indices['j'] = cell_ind_tuple[1]
            dict_cell_indices['i'] = cell_ind_tuple[2]
            cell_indices.append(dict_cell_indices)
                                
            # create new coordinate values for current cell
            cell_XC_0 = XC[tuple(np.asarray(cell_ind_tuple)[:,0])]
            mod_base = cell_XC_0 - 180
            cell_XC = np.nanmean(((XC[cell_ind_tuple] - mod_base) % 360) + mod_base) % 360
            XC_cell = np.hstack((XC_cell,cell_XC))
            YC_cell = np.hstack((YC_cell,np.nanmean(YC[cell_ind_tuple])))
            XG_cell = np.hstack((XG_cell,XG[tile,s_j,s_i]))
            YG_cell = np.hstack((YG_cell,YG[tile,s_j,s_i]))
            rA_cell = np.hstack((rA_cell,np.nansum(rA[cell_ind_tuple])))
            
            array_indexing[cell_ind_tuple] = int(len(XC_cell) - 1)
            incell_order[cell_ind_tuple] = np.arange(len(cell_ind_tuple[0]))
            already_covered[cell_ind_tuple] = 1
    
    return cell_indices,array_indexing,incell_order,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell,already_covered
    

def nested_geom_create(list_nested,XC,YC,XG,YG,rA,already_covered=None):
    """
    Creates different levels of nested geometry to simplify covariance calculations.
    Input list_nested is a list, with each entry containing a dictionary for a given cell merging scheme.
    So if in the new geometry we want some cells to stay 1x1, to merge some cells 3x3, and others 10x10,
    there would be 3 entries in the list.
    The dictionary then has keys 'n_merge' and 'tile';
    the value of 'n_merge' is a positive integer indicating the number of cells to merge 
    (1, 3, and 10 respectively in the above example).
    the value of 'tile' is a dictionary containing key: integer in the range 0-12
    and value: dictionary indicating the 'j' and 'i' indices in the tile to apply the merging scheme to.
    Multiple j,i ranges can be included in a given tile by specifying value: list containing the dictionaries.
    If a region is already nested in an earlier entry of list_nested, it will be excluded in
    the tiling scheme for the current list_nested entry.
    Optional input already_covered is a NumPy array 13x90x90 with ones indicating cells already included
    in a cell grouping, and zeros otherwise.    
    
    cell_indices is a list with dictionaries containing the tile,j,i indices of ECCO grid cells contained
    in that output cell.
    array_indexing is a NumPy array indicating the output cell number for each input grid cell.
    incell_order is a NumPy array indicating the order of each input grid cell within the cell.
    New coordinate NumPy vectors XC_cell,YC_cell,XG_cell,YG_cell,rA_cell are also generated for the cells.
    """
    
    cell_indices = []
    XC_cell = np.empty((0,))
    YC_cell = np.empty((0,))
    XG_cell = np.empty((0,))
    YG_cell = np.empty((0,))
    rA_cell = np.empty((0,))
    array_indexing = np.empty(XC.shape,dtype=np.int64)
    incell_order = np.empty(XC.shape,dtype=np.int64)
    if isinstance(already_covered,np.ndarray) == False:
        if already_covered == None:
            already_covered = np.zeros(XC.shape)
    for curr_nest in list_nested:
        curr_n_merge = curr_nest['n_merge']
        curr_tiles = curr_nest['tile']
        for tile,indices in curr_tiles.items():
            if isinstance(indices,list):
                for curr_indices in indices:
                    cell_indices,array_indexing,incell_order,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell,already_covered = \
                                                nested_geom_attile(tile,curr_indices,curr_n_merge,\
        cell_indices,array_indexing,incell_order,\
        XC_cell,YC_cell,XG_cell,YG_cell,rA_cell,\
        XC,YC,XG,YG,rA,\
        already_covered)
            else:
                cell_indices,array_indexing,incell_order,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell,already_covered = \
                                            nested_geom_attile(tile,indices,curr_n_merge,\
        cell_indices,array_indexing,incell_order,\
        XC_cell,YC_cell,XG_cell,YG_cell,rA_cell,\
        XC,YC,XG,YG,rA,\
                                                               already_covered)
    
    return cell_indices,array_indexing,incell_order,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell



def dir_line(lon_0,lon_1,lat_0,lat_1):
    dir_seg = np.angle((np.cos((np.pi/180)*((0.5*lat_0) + (0.5*lat_1)))\
                         *((((lon_1 - lon_0) + 180) % 360) - 180))\
                         + (1j*(lat_1 - lat_0)))

    return dir_seg


def cells_along_coast(lon_coast_pts,lat_coast_pts,dist_offshore_km,direction_offshore,XC,YC):

    dirs_seg = np.array([])
    dirs_offshore = np.array([])
    for lon_seg_endpts,lat_seg_endpts in zip(zip(lon_coast_pts[:-1],lon_coast_pts[1:]),\
                                         zip(lat_coast_pts[:-1],lat_coast_pts[1:])):
        dir_seg = dir_line(lon_seg_endpts[0],lon_seg_endpts[1],\
                           lat_seg_endpts[0],lat_seg_endpts[1])        
        if direction_offshore == 'left':
            dir_offshore = dir_seg + (np.pi/2)
        elif direction_offshore == 'right':
            dir_offshore = dir_seg - (np.pi/2)
        dirs_seg = np.append(dirs_seg,dir_seg)
        dirs_offshore = np.append(dirs_offshore,dir_offshore)

    in_offshore_region_flat_ind = np.array([]).astype('int64')
    for seg_count,(lon_seg_endpts,lat_seg_endpts,dir_seg,dir_offshore)\
            in enumerate(zip(zip(lon_coast_pts[:-1],lon_coast_pts[1:]),\
                         zip(lat_coast_pts[:-1],lat_coast_pts[1:]),\
                         dirs_seg,dirs_offshore)):

        lon_seg_endpts_offshore = (np.cos(dir_offshore)\
                                    *(dist_offshore_km/(111.1*np.cos((np.pi/180)*np.asarray(lat_seg_endpts)))))\
                                    + np.asarray(lon_seg_endpts)
        lat_seg_endpts_offshore = (np.sin(dir_offshore)*(dist_offshore_km/111.1)) + np.asarray(lat_seg_endpts)
        dir_from_coast_0 = dir_line(np.array([[[lon_seg_endpts[0]]]]),XC,\
                                    np.array([[[lat_seg_endpts[0]]]]),YC)
        dir_from_coast_1 = dir_line(np.array([[[lon_seg_endpts[1]]]]),XC,\
                                    np.array([[[lat_seg_endpts[1]]]]),YC)
        dir_from_offshore_0 = dir_line(np.array([[[lon_seg_endpts_offshore[0]]]]),XC,\
                                       np.array([[[lat_seg_endpts_offshore[0]]]]),YC)
        dir_from_offshore_1 = dir_line(np.array([[[lon_seg_endpts_offshore[1]]]]),XC,\
                                       np.array([[[lat_seg_endpts_offshore[1]]]]),YC)
        
        if direction_offshore == 'left':
            if ((seg_count > 0) and \
              (((dir_seg - dirs_seg[seg_count-1] + np.pi) % (2*np.pi)) - np.pi < 0)):
                # include regions near turning points on the coastline
                in_rad_turn = ((dir_from_coast_0 - dir_offshore) % (2*np.pi) \
                               <= (dirs_offshore[seg_count-1] - dir_offshore) % (2*np.pi))
                dist_from_coast_0 = 111.1*np.abs((np.cos((np.pi/180)*lat_seg_endpts[0])*\
                                                  (((XC - lon_seg_endpts[0] + 180) % 360) - 180))\
                                                 + (1j*(YC - lat_seg_endpts[0])))
                in_offshore_region_flat_ind = np.unique(np.append(in_offshore_region_flat_ind,\
                                                         np.logical_and(in_rad_turn,\
                                                                        dist_from_coast_0 <= dist_offshore_km)\
                                                                 .flatten().nonzero()))

            # masks for along-coast regions
            in_coast_0_quad = ((dir_from_coast_0 - dir_seg) % (2*np.pi) <= (np.pi/2))
            in_coast_1_quad = ((dir_from_coast_1 - (dir_seg + np.pi)) % (2*np.pi) >= ((3/2)*np.pi))
            in_offshore_0_quad = ((dir_from_offshore_0 - dir_seg) % (2*np.pi) >= ((3/2)*np.pi))
            in_offshore_1_quad = ((dir_from_offshore_1 - (dir_seg + np.pi)) % (2*np.pi) <= (np.pi/2))
        elif direction_offshore == 'right':
            if ((seg_count > 0) and \
              (((dir_seg - dirs_seg[seg_count-1] + np.pi) % (2*np.pi)) - np.pi > 0)):
                # include regions near turning points on the coastline
                in_rad_turn = ((dir_from_coast_0 - dirs_offshore[seg_count-1]) % (2*np.pi) \
                               <= (dir_offshore - dirs_offshore[seg_count-1]) % (2*np.pi))
                dist_from_coast_0 = 111.1*np.abs((np.cos((np.pi/180)*lat_seg_endpts[0])*\
                                                  (((XC - lon_seg_endpts[0] + 180) % 360) - 180))\
                                                 + (1j*(YC - lat_seg_endpts[0])))
                in_offshore_region_flat_ind = np.unique(np.append(in_offshore_region_flat_ind,\
                                                         np.logical_and(in_rad_turn,\
                                                                        dist_from_coast_0 <= dist_offshore_km)\
                                                                 .flatten().nonzero()))

            in_coast_0_quad = ((dir_from_coast_0 - dir_seg) % (2*np.pi) >= (1.49*np.pi))
            in_coast_1_quad = ((dir_from_coast_1 - (dir_seg + np.pi)) % (2*np.pi) <= 0.51*np.pi) 
            in_offshore_0_quad = ((dir_from_offshore_0 - dir_seg) % (2*np.pi) <= 0.51*np.pi)
            in_offshore_1_quad = ((dir_from_offshore_1 - (dir_seg + np.pi)) % (2*np.pi) >= 1.49*np.pi)           

        in_offshore_region_flat_ind = np.unique(np.append(in_offshore_region_flat_ind,\
                                                    np.logical_and(np.logical_and(in_coast_0_quad,in_coast_1_quad),\
                                                    np.logical_and(in_offshore_0_quad,in_offshore_1_quad))\
                                                        .flatten().nonzero()))

    in_offshore_region_cell_ind = np.unravel_index(in_offshore_region_flat_ind,XC.shape)
    
    return in_offshore_region_cell_ind


def coast_coords_unsorted(XG_2d,YG_2d,maskC_2d,lon_bounds,lat_bounds):
    lon_midj_face = XG_2d[:-1,:] + ((((np.diff(XG_2d,axis=-2) + 180) % 360) - 180)/2)
    lat_midj_face = YG_2d[:-1,:] + (np.diff(YG_2d,axis=-2)/2)
    lon_midi_face = XG_2d[:,:-1] + ((((np.diff(XG_2d,axis=-1) + 180) % 360) - 180)/2)
    lat_midi_face = YG_2d[:,:-1] + (np.diff(YG_2d,axis=-1)/2)

    midj_in_region_ind = np.logical_and((lon_midj_face - lon_bounds[0]) % 360 <= np.diff(lon_bounds) % 360,\
                                        np.logical_and(lat_midj_face >= lat_bounds[0],\
                                                       lat_midj_face <= lat_bounds[1])).nonzero()
    midi_in_region_ind = np.logical_and((lon_midi_face - lon_bounds[0]) % 360 <= np.diff(lon_bounds) % 360,\
                                        np.logical_and(lat_midi_face >= lat_bounds[0],\
                                                       lat_midi_face <= lat_bounds[1])).nonzero()
    midj_in_region_j = np.unique(midj_in_region_ind[-2])
    midj_in_region_i = np.unique(midj_in_region_ind[-1])
    midi_in_region_j = np.unique(midi_in_region_ind[-2])
    midi_in_region_i = np.unique(midi_in_region_ind[-1])

    lon_midj_in_region = lon_midj_face[np.reshape(midj_in_region_j,(-1,1)),midj_in_region_i]
    lat_midj_in_region = lat_midj_face[np.reshape(midj_in_region_j,(-1,1)),midj_in_region_i]
    lon_midi_in_region = lon_midi_face[np.reshape(midi_in_region_j,(-1,1)),midi_in_region_i]
    lat_midi_in_region = lat_midi_face[np.reshape(midi_in_region_j,(-1,1)),midi_in_region_i]
    maskC_midj = np.diff(maskC_2d,axis=-1)\
                    [np.reshape(midj_in_region_j,(-1,1)),midj_in_region_i-1]
    maskC_midi = np.diff(maskC_2d,axis=-2)\
                    [np.reshape(midi_in_region_j,(-1,1))-1,midi_in_region_i]
    lon_coast_points = np.hstack((lon_midj_in_region[np.abs(maskC_midj) > 0.5],\
                                  lon_midi_in_region[np.abs(maskC_midi) > 0.5]))
    lat_coast_points = np.hstack((lat_midj_in_region[np.abs(maskC_midj) > 0.5],\
                                  lat_midi_in_region[np.abs(maskC_midi) > 0.5]))

    return lon_coast_points,lat_coast_points


def remove_already_covered_cells(cells_tup,already_covered_array):
    # remove already covered cells and update already_covered_array
    not_already_covered = (already_covered_array[cells_tup] < 0.5).nonzero()[0]
    cells_tup = tuple(np.asarray(cells_tup)[:,not_already_covered])
    already_covered_array[cells_tup] = 1

    return cells_tup,already_covered_array


def ind_dict_coords_create(cell_tup,XC,YC,rA,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell):
    cell_ind_dict = {'tile':cell_tup[0],\
                     'j':cell_tup[1],\
                     'i':cell_tup[2]}
    
    # create new coordinate values for current cell
    cell_XC_0 = XC[tuple(np.asarray(cell_tup)[:,0])]
    mod_base = cell_XC_0 - 180
    cell_XC = np.nanmean(((XC[cell_tup] - mod_base) % 360) + mod_base) % 360
    XC_cell = np.hstack((XC_cell,cell_XC))
    YC_cell = np.hstack((YC_cell,np.nanmean(YC[cell_tup])))    
    XG_cell = np.hstack((XG_cell,np.array([np.nan])))
    YG_cell = np.hstack((YG_cell,np.array([np.nan])))
    rA_cell = np.hstack((rA_cell,np.nansum(rA[cell_tup])))
    
    return cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell



def Gulf_NAtl_regional_scheme(ds_grid):
    """Define cell indexing for regions in the Gulf/Carib/mid-latitude N Atlantic"""

    cell_indices = []
    XC_cell = np.empty((0,))
    YC_cell = np.empty((0,))
    XG_cell = np.empty((0,))
    YG_cell = np.empty((0,))
    rA_cell = np.empty((0,))
    
    # identify cells in specifically defined regions
    
    array_indexing = np.empty((ds_grid.XC.shape),dtype=np.int64)
    array_indexing.fill(-1000)
    already_covered = np.zeros((ds_grid.XC.shape))
    already_covered[ds_grid.maskC.isel(k=0).values < 0.5] = 1
    
    # # Mississippi discharge
    cells_Miss = np.logical_and(((ds_grid.XC.values - (-90)) % 360) < 1,\
                                np.logical_and(ds_grid.YC.values >= 28,ds_grid.YC.values <= 29.7)).nonzero()
    cells_Miss,already_covered = remove_already_covered_cells(cells_Miss,already_covered)
    array_indexing[cells_Miss] = 0
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_Miss,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)
    
    # # East Gulf
    cells_EGulf = np.logical_and(((ds_grid.XC.values - (-89)) % 360) < 7,\
                                 np.logical_and(ds_grid.YC.values >= 22,ds_grid.YC.values <= 31)).nonzero()
    cells_EGulf,already_covered = remove_already_covered_cells(cells_EGulf,already_covered)
    array_indexing[cells_EGulf] = 1
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_EGulf,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)

    # # West Gulf
    cells_WGulf = np.logical_and(((ds_grid.XC.values - (-98)) % 360) < 9,\
                                 np.logical_and(ds_grid.YC.values >= 18,ds_grid.YC.values <= 30)).nonzero()
    cells_WGulf,already_covered = remove_already_covered_cells(cells_WGulf,already_covered)
    array_indexing[cells_WGulf] = 2
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_WGulf,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)

    # # Caribbean
    cells_Carib = np.logical_and(((ds_grid.XC.values - (-89)) % 360) < 28,\
                                 np.logical_and(ds_grid.YC.values >= 9,ds_grid.YC.values < 22)).nonzero()
    # remove Pacific
    noPac = (((2/3)*(ds_grid.XC.values[cells_Carib] - (-83))) + (ds_grid.YC.values[cells_Carib] - 10) >= 0)\
                .flatten().nonzero()[0]
    cells_Carib = tuple(np.asarray(cells_Carib)[:,noPac])
    # remove (open) Atlantic
    noAtl = (((1/4)*(ds_grid.XC.values[cells_Carib] - (-72))) + (ds_grid.YC.values[cells_Carib] - 20) <= 0)\
                .flatten().nonzero()[0]
    cells_Carib = tuple(np.asarray(cells_Carib)[:,noAtl])
    cells_Carib,already_covered = remove_already_covered_cells(cells_Carib,already_covered)
    array_indexing[cells_Carib] = 3
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_Carib,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)

    # # open Atlantic: 5-25 N
    cells_Atl_525N = np.logical_and(((ds_grid.XC.values + ((3/23)*ds_grid.YC.values) - (-76)) % 360) < 79,\
                                     np.logical_and(ds_grid.YC.values >= 5,ds_grid.YC.values <= 25)).nonzero()
    cells_Atl_525N,already_covered = remove_already_covered_cells(cells_Atl_525N,already_covered)
    array_indexing[cells_Atl_525N] = 4
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_Atl_525N,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)

    # # Florida Straits
    cells_FLstr = np.logical_and(((ds_grid.XC.values - (-82)) % 360) < 3,\
                                 np.logical_and(ds_grid.YC.values >= 22,ds_grid.YC.values <= 27)).nonzero()
    cells_FLstr,already_covered = remove_already_covered_cells(cells_FLstr,already_covered)
    array_indexing[cells_FLstr] = 5
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_FLstr,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)

    # # US Atlantic coast
    lon_bounds = np.array([-81.5,-66.4])
    lat_bounds = np.array([27,45])
    lon_coast_points,lat_coast_points = coast_coords_unsorted(ds_grid.XG.isel(tile=10).values,\
                                                              ds_grid.YG.isel(tile=10).values,\
                                                              ds_grid.maskC.isel(k=0,tile=10).values,\
                                                              lon_bounds,lat_bounds)
    # remove Bahamas
    closer_to_coast = ((-1*(lon_coast_points - (-79))) + (lat_coast_points - 25) >= 0).nonzero()[0]
    # order by increasing latitude
    coast_order = np.argsort(lat_coast_points[closer_to_coast] + (.001*lon_coast_points[closer_to_coast]))
    lon_coast_points = lon_coast_points[closer_to_coast[coast_order]]
    lat_coast_points = lat_coast_points[closer_to_coast[coast_order]]
    # find cells near coast
    cells_USAtl = cells_along_coast(lon_coast_points,lat_coast_points,300,'right',\
                                    ds_grid.XC.values,ds_grid.YC.values)
    cells_USAtl,already_covered = remove_already_covered_cells(cells_USAtl,already_covered)
    array_indexing[cells_USAtl] = 6
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_USAtl,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)


    # # open Atlantic: 25-35 N
    cells_Atl_2535N = np.logical_and(((ds_grid.XC.values - (-80)) % 360) < 80,\
                                     np.logical_and(ds_grid.YC.values >= 25,ds_grid.YC.values <= 35)).nonzero()
    cells_Atl_2535N,already_covered = remove_already_covered_cells(cells_Atl_2535N,already_covered)
    array_indexing[cells_Atl_2535N] = 7
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_Atl_2535N,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)

    # # open Atlantic: 35-45 N
    cells_Atl_3545N = np.logical_and(((ds_grid.XC.values - (-80)) % 360) < 80,\
                                     np.logical_and(ds_grid.YC.values > 35,ds_grid.YC.values <= 45)).nonzero()
    # remove Mediterranean
    noMed = (((-1.4)*(ds_grid.XC.values[cells_Atl_3545N] - (-5.5))) + (ds_grid.YC.values[cells_Atl_3545N] - 36) >= 0)\
                .flatten().nonzero()[0]
    cells_Atl_3545N = tuple(np.asarray(cells_Atl_3545N)[:,noMed])
    cells_Atl_3545N,already_covered = remove_already_covered_cells(cells_Atl_3545N,already_covered)
    array_indexing[cells_Atl_3545N] = 8
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_Atl_3545N,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)

    # # Atlantic: 45-55 N
    cells_Atl_4555N = np.logical_and(((ds_grid.XC.values - (-75)) % 360)\
                         - (75 + ((-0.8)*(ds_grid.YC.values - 53.125))) < 0,\
                                     np.logical_and(ds_grid.YC.values > 45,ds_grid.YC.values <= 55)).nonzero()
    cells_Atl_4555N,already_covered = remove_already_covered_cells(cells_Atl_4555N,already_covered)
    array_indexing[cells_Atl_4555N] = 9
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_Atl_4555N,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)

    # # Atlantic: 55-70 N
    cells_Atl_5570N = np.logical_and(((ds_grid.XC.values - (-73)) % 360) < 93,\
                                     np.logical_and(ds_grid.YC.values > 55,ds_grid.YC.values <= 70)).nonzero()
    cells_Atl_5570N,already_covered = remove_already_covered_cells(cells_Atl_5570N,already_covered)
    array_indexing[cells_Atl_5570N] = 10
    cell_ind_dict,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell = \
                    ind_dict_coords_create(cells_Atl_5570N,ds_grid.XC.values,ds_grid.YC.values,ds_grid.rA.values,\
                                           XC_cell,YC_cell,XG_cell,YG_cell,rA_cell)
    cell_indices.append(cell_ind_dict)
    
    return cell_indices,array_indexing,already_covered,XC_cell,YC_cell,XG_cell,YG_cell,rA_cell