# filtering and interpolation functions

def bandpass_err_fcn(input_array,axis=0,delta_dim=1,low_bound=1.e-15,high_bound=1.e30,power_cutoff_opt=1,steepness_factor=5,trend_handling_opt=1,nan_handling_opt=1,interp_opt=1,edge_handling_opt=0,uneven_edge_handling_opt=1):
    """Apply error function-based bandpass filters to input_array.

    This function applies error function-based bandpass filters to input_array along specified axis.
    Other input parameters include
    delta_dim: spacing between adjacent coordinates (points) along axis
    low_bound: frequency/wavenumber of low cutoff
    high_bound: frequency/wavenumber of high cutoff
    power_cutoff_opt: 0 = cutoff is at half-amplitude of filter, 1 = cutoff is at half-power
    steepness_factor: determines steepness of filter near cutoff
    trend_handling_opt: 0 = output is detrended, 1 = output includes trend
    nan_handling_opt: 0 = NaNs are zeroed, 1 = NaNs are included
    interp_opt: 0 = replace NaNs with zeros for filtering, 1 = interpolate NaNs internally for filtering
    edge_handling_opt: 0 = edges are not masked out, 1 = edges are masked out, as a function of dominant scales in input_array
    uneven_edge_handling_opt: 0 = edges are the ends of the array, 1 = edges are adjusted to exclude NaNs adjacent to edges of array

    Outputs included in list_of_outputs (all numpy arrays):
    [0]: bandpassed_array, input array with low- and high-pass error function filters applied along dimension axis
    [1]: trend_array, same size/shape as bandpassed_array, giving the linear regression trend along dimension axis"""
    
    pass

    import numpy as np
    from scipy import special
    
    
    orig_array_shape = input_array.shape
    if len(orig_array_shape) == 1:
        input_array = np.reshape(input_array,(1,-1))
        if axis == 0:
             axis = -1


    dim_length = input_array.shape[axis]

    f_vec = np.reshape(((1/(delta_dim*(2*dim_length)))*((((np.r_[0:(2*dim_length)]) + dim_length) % (2*dim_length)) - dim_length)),(1,-1))
    n_avg_ind = (np.round(((1/8)*np.exp(-np.mean(np.log(np.array([np.fmax(low_bound,1/(delta_dim*dim_length)),np.fmin(high_bound,1/(2*delta_dim))])))))/delta_dim)).astype('i8')

    if power_cutoff_opt == 1:
        half_power_adj = np.exp(special.erfinv((2**(1/2)) - 1)/steepness_factor)
    else:
        half_power_adj = 1
    bandpass_filter = 0.5*(special.erf(steepness_factor*(np.log(np.abs(f_vec)) - np.log(low_bound/half_power_adj))) - special.erf(steepness_factor*(np.log(np.abs(f_vec)) - np.log(high_bound*half_power_adj))))

    padding_ind_inside = (np.floor(dim_length/2) + (np.r_[0:dim_length])).astype('i8')
    if uneven_edge_handling_opt == 0:
        padding_ind_begin = (np.r_[0:np.floor(dim_length/2)]).astype('i8')
        padding_ind_end = (np.r_[(np.floor(dim_length/2) + dim_length):(2*dim_length)]).astype('i8')

    size_array = np.array(input_array.shape)
    prod_size_not_inc_bp_dim = np.prod(np.delete(size_array,axis))

    input_array_permuted = np.reshape(np.moveaxis(input_array,axis,-1),(prod_size_not_inc_bp_dim,dim_length))
    input_array_masked = np.ma.array(input_array_permuted,mask=np.asarray(np.logical_or(np.logical_or(np.isnan(input_array_permuted),np.isinf(input_array_permuted)),np.abs(input_array_permuted) < (1.e-15)*np.nanmax(np.abs(input_array_permuted)))))
    # remove means and trends
    index_array_tiled = np.tile(np.r_[0:dim_length],(prod_size_not_inc_bp_dim,1))
    index_array_nomean = np.ma.array(index_array_tiled - np.tile((np.ma.array(index_array_tiled,mask=(input_array_masked).mask)).mean(axis=-1,keepdims=1),(1,dim_length)),mask=(input_array_masked).mask)
    input_array_mean_tiled = np.tile((np.ma.mean(input_array_masked,axis=-1,keepdims=1)).data,(1,dim_length))
    input_array_nomean = input_array_permuted - input_array_mean_tiled
    trend_array_permuted = input_array_mean_tiled + (((index_array_nomean.data)*np.tile((1/((index_array_nomean**2).sum(axis=-1,keepdims=1)))*((index_array_nomean*input_array_nomean).sum(axis=-1,keepdims=1)),(1,dim_length))).data)
    trend_array_permuted[np.tile((np.sum(~(input_array_masked.mask),axis=-1,keepdims=1) < 0.5),(1,dim_length))] = np.nan

    input_array_permuted_minus_trend = np.ma.array(input_array_permuted - trend_array_permuted,mask=(input_array_masked).mask,fill_value=0)

    input_padded = np.zeros((prod_size_not_inc_bp_dim,(2*dim_length)))

    if edge_handling_opt == 1:
        test_lags = np.concatenate((np.array([1,2,3]),(np.round(np.exp(np.r_[1.5:np.log(dim_length):0.3])).astype('i8'))))
        N = len(test_lags)
        autocorr_at_lags = np.empty((prod_size_not_inc_bp_dim,N))
        autocorr_at_lags.fill(np.nan)
        integral_scale = np.empty((prod_size_not_inc_bp_dim,1))
        integral_scale.fill(np.nan)
        n_lag_step = 0
        complete_flag = 0
        while ((complete_flag == 0) and (n_lag_step < N)):
            lag_ind = test_lags[n_lag_step]
            part_1_ind = (np.r_[0:(dim_length - lag_ind)]).astype('i8')
            part_2_ind = (np.r_[lag_ind:dim_length]).astype('i8')

            mask_combined = np.logical_or((input_array_permuted_minus_trend[:,part_1_ind]).mask,(input_array_permuted_minus_trend[:,part_2_ind]).mask)
            part_1_mean = (np.ma.array(input_array_permuted_minus_trend[:,part_1_ind],mask=mask_combined)).mean(axis=-1,keepdims=1)
            part_2_mean = (np.ma.array(input_array_permuted_minus_trend[:,part_2_ind],mask=mask_combined)).mean(axis=-1,keepdims=1)
            part_1_corr = np.ma.array(input_array_permuted_minus_trend[:,part_1_ind] - np.tile(part_1_mean,(1,dim_length - lag_ind)),mask=mask_combined)
            part_2_corr = np.ma.array(input_array_permuted_minus_trend[:,part_2_ind] - np.tile(part_2_mean,(1,dim_length - lag_ind)),mask=mask_combined)
            autocorr_at_lags[:,n_lag_step] = (np.ma.sum(part_1_corr*part_2_corr,axis=-1))/(((np.ma.sum(part_1_corr**2,axis=-1))**(1/2))*((np.ma.sum(part_2_corr**2,axis=-1))**(1/2)))
            autocorr_at_lags[np.sum(~mask_combined,axis=-1) < 0.5,n_lag_step] = np.nan
            if n_lag_step == N - 1:
                first_neg = np.reshape(np.isnan(integral_scale),(-1,1))
            else:
                first_neg = np.reshape(np.logical_and(np.reshape(autocorr_at_lags[:,n_lag_step] < 0,(-1,1)),np.reshape(np.isnan(integral_scale),(-1,1))),(-1,1))
            if np.sum(first_neg) > 0:                
                if n_lag_step == 0:
                    integral_scale[first_neg] = delta_dim
                else:
                    first_neg_mask = np.tile(first_neg,(1,n_lag_step + 1))
                    curr_integral_scale = delta_dim*(np.reshape(first_neg_mask[:,0],(-1,1)) + (2*np.sum(np.tile(0.5*(np.reshape(test_lags[np.concatenate((np.r_[1:(n_lag_step + 1)],np.array([n_lag_step])))] - np.concatenate((np.array([0]),test_lags[0:n_lag_step])),(1,-1))),(prod_size_not_inc_bp_dim,1))*first_neg_mask*autocorr_at_lags[:,0:(n_lag_step + 1)],axis=-1,keepdims=1)))
                    integral_scale[first_neg] = curr_integral_scale[first_neg]
            if np.sum(np.isnan(integral_scale)) == 0:
                complete_flag = 1
            n_lag_step = n_lag_step + 1
        freq_dominant_approx = 1/(np.pi*integral_scale)
    # interpolate to replace internal NaNs
    input_array_permuted_notrend = input_array_permuted_minus_trend.data
    if interp_opt == 1:
        for row_ind in range(prod_size_not_inc_bp_dim):
            curr_good_ind = (~(input_array_masked[row_ind,:]).mask).nonzero()[0]
            if len(curr_good_ind) > 0.5:
                input_array_permuted_notrend[row_ind,:] = np.interp(np.r_[0:dim_length],curr_good_ind,(input_array_permuted_minus_trend[row_ind,curr_good_ind]).data,left=np.nan,right=np.nan)
    input_padded[:,padding_ind_inside] = input_array_permuted_notrend

    # pad with erf slopes on either end of the range
    if uneven_edge_handling_opt == 1:
        first_good_ind = np.ones((prod_size_not_inc_bp_dim,1))
        last_good_ind = dim_length*np.ones((prod_size_not_inc_bp_dim,1))
        cum_forward_nan_mask = np.cumsum(~(input_array_masked.mask),axis=-1)
        cum_reverse_nan_mask = np.flip(np.cumsum(np.flip(~(input_array_masked.mask),axis=-1),axis=-1),axis=-1)
        dim_ind_array = np.tile(np.reshape(np.r_[0:dim_length],(1,-1)),(prod_size_not_inc_bp_dim,1))
        first_good_mask = np.logical_and(cum_forward_nan_mask > 1.e-5,np.concatenate((np.zeros((prod_size_not_inc_bp_dim,1)),cum_forward_nan_mask[:,:-1]),axis=-1) < 1.e-5)
        first_good_ind[cum_forward_nan_mask[:,-1] > 1.e-5] = np.sum(first_good_mask*dim_ind_array,axis=-1,keepdims=1)[cum_forward_nan_mask[:,-1] > 1.e-5]
        last_good_mask = np.logical_and(cum_reverse_nan_mask > 1.e-5,np.concatenate((cum_reverse_nan_mask[:,1:],np.zeros((prod_size_not_inc_bp_dim,1))),axis=-1) < 1.e-5)
        last_good_ind[cum_reverse_nan_mask[:,0] > 1.e-5] = np.sum(last_good_mask*dim_ind_array,axis=-1,keepdims=1)[cum_reverse_nan_mask[:,0] > 1.e-5]
    if n_avg_ind > 0:
        if uneven_edge_handling_opt == 0:
            pad_slope_begin = (1/n_avg_ind)*(np.mean(input_padded[:,(np.floor(dim_length/2)).astype('i8') + np.r_[n_avg_ind:(2*n_avg_ind)]],axis=-1,keepdims=1) - np.mean(input_padded[:,(np.floor(dim_length/2)).astype('i8') + np.r_[0:n_avg_ind]],axis=-1,keepdims=1))
            pad_val_begin = np.mean(input_padded[:,(np.floor(dim_length/2)).astype('i8') + np.r_[0:n_avg_ind]],axis=-1,keepdims=1) - (((n_avg_ind - 1)/2)*pad_slope_begin)
            pad_slope_end = (1/n_avg_ind)*(np.mean(input_padded[:,(np.floor(dim_length/2)).astype('i8') + dim_length + np.r_[(-n_avg_ind):0]],axis=-1,keepdims=1) - np.mean(input_padded[:,(np.floor(dim_length/2)).astype('i8') + dim_length + np.r_[(-2*n_avg_ind):(-n_avg_ind)]],axis=-1,keepdims=1))
            pad_val_end = np.mean(input_padded[:,(np.floor(dim_length/2)).astype('i8') + dim_length + np.r_[(-n_avg_ind):0]],axis=-1,keepdims=1) + (((n_avg_ind - 1)/2)*pad_slope_end)
        else:
            dim_ind_array = np.tile(np.reshape(np.r_[0:(2*dim_length)],(1,-1)),(prod_size_not_inc_bp_dim,1))
            dist_from_first_good = dim_ind_array - ((np.floor(dim_length/2)).astype('i8') + np.tile(first_good_ind,(1,2*dim_length)))
            avg_range_mask_1 = np.logical_and(dist_from_first_good >= 0,dist_from_first_good < n_avg_ind)
            avg_range_mask_2 = np.logical_and(dist_from_first_good >= n_avg_ind,dist_from_first_good < (2*n_avg_ind))
            pad_slope_begin = (1/n_avg_ind)*((np.sum(avg_range_mask_2*input_padded,axis=-1,keepdims=1)/np.sum(avg_range_mask_2,axis=-1,keepdims=1)) - (np.sum(avg_range_mask_1*input_padded,axis=-1,keepdims=1)/np.sum(avg_range_mask_1,axis=-1,keepdims=1)))
            pad_val_begin = (np.sum(avg_range_mask_1*input_padded,axis=-1,keepdims=1)/np.sum(avg_range_mask_1,axis=-1,keepdims=1)) - (((n_avg_ind - 1)/2)*pad_slope_begin)
            dist_from_last_good = dim_ind_array - ((np.floor(dim_length/2)).astype('i8') + np.tile(last_good_ind,(1,2*dim_length)))
            avg_range_mask_1 = np.logical_and(dist_from_last_good > -n_avg_ind,dist_from_last_good <= 0)
            avg_range_mask_2 = np.logical_and(dist_from_last_good > -(2*n_avg_ind),dist_from_last_good <= -n_avg_ind)
            pad_slope_end = (1/n_avg_ind)*(-((np.sum(avg_range_mask_2*input_padded,axis=-1,keepdims=1)/np.sum(avg_range_mask_2,axis=-1,keepdims=1)) - (np.sum(avg_range_mask_1*input_padded,axis=-1,keepdims=1)/np.sum(avg_range_mask_1,axis=-1,keepdims=1))))
            pad_val_end = (np.sum(avg_range_mask_1*input_padded,axis=-1,keepdims=1)/np.sum(avg_range_mask_1,axis=-1,keepdims=1)) + (((n_avg_ind - 1)/2)*pad_slope_end)
    else:
        if uneven_edge_handling_opt == 0:
            pad_slope_begin = np.reshape(input_padded[:,(np.floor(dim_length/2)).astype('i8')],(-1,1))/0.5
            pad_val_begin = np.reshape(input_padded[:,(np.floor(dim_length/2)).astype('i8')],(-1,1))
            pad_slope_end = np.reshape(input_padded[:,(np.floor(dim_length/2)).astype('i8') + dim_length - 1],(-1,1))/0.5
            pad_val_end = np.reshape(input_padded[:,(np.floor(dim_length/2)).astype('i8') + dim_length - 1],(-1,1))
        else:
            dim_ind_array = np.tile(np.reshape(np.r_[0:(2*dim_length)],(1,-1)),(prod_size_not_inc_bp_dim,1))
            avg_range_mask = (np.abs(dim_ind_array - ((np.floor(dim_length/2)).astype('i8') + np.tile(first_good_ind,(1,2*dim_length)))) < 0.5)
            pad_slope_begin = (np.sum(avg_range_mask*input_padded,axis=-1,keepdims=1)/np.sum(avg_range_mask,axis=-1,keepdims=1))/0.5
            pad_val_begin = np.sum(avg_range_mask*input_padded,axis=-1,keepdims=1)/np.sum(avg_range_mask,axis=-1,keepdims=1)
            avg_range_mask = (np.abs(dim_ind_array - ((np.floor(dim_length/2)).astype('i8') + np.tile(last_good_ind,(1,2*dim_length)))) < 0.5)
            pad_slope_end = (np.sum(avg_range_mask*input_added,axis=-1,keepdims=1)/np.sum(avg_range_mask,axis=-1,keepdims=1))/0.5
            pad_val_end = np.sum(avg_range_mask*input_padded,axis=-1,keepdims=1)/np.sum(avg_range_mask,axis=-1,keepdims=1)
    if uneven_edge_handling_opt == 0:
        input_padded[:,padding_ind_begin] = np.tile(pad_val_begin,(1,len(padding_ind_begin)))*(1 + special.erf(((np.pi**(1/2))/2)*np.tile(np.abs(pad_slope_begin/pad_val_begin),(1,len(padding_ind_begin)))*(np.tile(np.reshape(padding_ind_begin,(1,-1)),(prod_size_not_inc_bp_dim,1)) - np.nanmin(padding_ind_inside))))
        input_padded[:,padding_ind_end] = np.tile(pad_val_end,(1,len(padding_ind_end)))*(1 - special.erf(((np.pi**(1/2))/2)*np.tile(np.abs(pad_slope_end/pad_val_end),(1,len(padding_ind_end)))*(np.tile(np.reshape(padding_ind_end,(1,-1)),(prod_size_not_inc_bp_dim,1)) - np.nanmax(padding_ind_inside))))
    else:
        begin_padding_array = np.tile(pad_val_begin,(1,2*dim_length))*(1 + special.erf(((np.pi**(1/2))/2)*np.tile(np.abs(pad_slope_begin/pad_val_begin),(1,2*dim_length))*dist_from_first_good))
        input_padded[dist_from_first_good < 0] = begin_padding_array[dist_from_first_good < 0]
        end_padding_array = np.tile(pad_val_end,(1,2*dim_length))*(1 - special.erf(((np.pi**(1/2))/2)*np.tile(np.abs(pad_slope_end/pad_val_end),(1,2*dim_length))*dist_from_last_good))
        input_padded[dist_from_last_good > 0] = end_padding_array[dist_from_last_good > 0]
    input_padded[np.logical_or(np.isnan(input_padded),np.isinf(input_padded))] = 0

    # apply filter in spectral domain
    bandpassed_array_padded = np.fft.ifft(np.tile(bandpass_filter,(prod_size_not_inc_bp_dim,1))*np.fft.fft(input_padded,axis=-1),axis=-1)

    mean_input_padded = np.mean(input_padded,axis=-1,keepdims=1)

    if trend_handling_opt == 0:
        bandpassed_array_not_padded = bandpassed_array_padded[:,padding_ind_inside]
    elif trend_handling_opt == 1:
        bandpassed_array_not_padded = bandpassed_array_padded[:,padding_ind_inside] + trend_array_permuted + np.tile(mean_input_padded,(1,dim_length))
    bandpassed_array_not_padded[input_array_masked.mask] = np.nan

    if edge_handling_opt == 1:
        denom_edge_scale = (delta_dim*high_bound)*((1 - ((low_bound/high_bound)**0.5))**1.25)
        n_edge_err = np.round(np.fmax(np.fmin(((((0.9*(low_bound/((np.pi/2)*freq_dominant_approx))) - (3/40))/(delta_dim*((((np.pi/2)*freq_dominant_approx)*low_bound)**(1/2)))) + ((0.45 - (1.8*(low_bound/((np.pi/2)*freq_dominant_approx))))/denom_edge_scale)),(0.15/(delta_dim*((((np.pi/2)*freq_dominant_approx)*low_bound)**(1/2))))),(0.3/np.tile(np.reshape(denom_edge_scale,(1,-1)),(prod_size_not_inc_bp_dim,1)))))
        dim_ind_array = np.tile(np.reshape(np.r_[0:dim_length],(1,-1)),(prod_size_not_inc_bp_dim,1))
        if uneven_edge_handling_opt == 0:
            bandpassed_array_not_padded[dim_ind_array - np.tile(n_edge_err,(1,dim_length)) < -0.5] = np.nan
            bandpassed_array_not_padded[dim_ind_array - np.tile(dim_length - n_edge_err,(1,dim_length)) > -0.5] = np.nan
        else:
            bandpassed_array_not_padded[dim_ind_array - np.tile(first_good_ind + n_edge_err,(1,dim_length)) < -0.5] = np.nan
            bandpassed_array_not_padded[dim_ind_array - np.tile(last_good_ind - n_edge_err,(1,dim_length)) > 0.5] = np.nan

    bandpassed_array = np.real(np.moveaxis(np.reshape(bandpassed_array_not_padded,np.concatenate((np.delete(size_array,axis),np.array([dim_length])))),-1,axis))
    trend_array = np.moveaxis(np.reshape(trend_array_permuted,np.concatenate((np.delete(size_array,axis),np.array([dim_length])))),-1,axis)

    # replace NaNs with zeros, depending on option given
    if nan_handling_opt == 0:
        bandpassed_array[np.isnan(bandpassed_array)] = 0
    
    
    # reshape output arrays to 1-D if needed to match input array size
    if len(orig_array_shape) == 1:
        bandpassed_array = np.reshape(bandpassed_array,(-1,))
        trend_array = np.reshape(trend_array,(-1,))


    list_of_outputs = [bandpassed_array,trend_array]
    return list_of_outputs


#===============================================================================


def seasonal_cycle_harmonics(input_array,input_times,num_harmonics,seasonal_output_times=[],time_axis_num=0):
    """
    Outputs input_array with seasonal cycle removed.
    If seasonal_output_times is not empty, also outputs the seasonal cycle on specified time points as second argument.
    """
    
    pass
    
    import numpy as np
    
    input_array_moveaxis = np.moveaxis(input_array,time_axis_num,0)
    input_array_2d = np.reshape(input_array_moveaxis,(input_array.shape[time_axis_num],-1))
    mask = np.asarray(np.logical_and(np.logical_and(~np.isnan(input_array_2d),~np.isinf(input_array_2d)),\
                                     np.abs(input_array_2d) > 1e-15))
    mask_sum = np.sum(mask,axis=0,keepdims=1)
    
    time_values = np.reshape(((input_times - np.datetime64('2006-01-01','ns')).astype('float64'))/(8.64e13),(-1,1))
    if len(seasonal_output_times) > 0:
        seasonal_output_tvalues = np.reshape(((seasonal_output_times - np.datetime64('2006-01-01','ns'))\
                                                  .astype('float64'))/(8.64e13),(-1,1))
    array_mean_2d = np.nansum(mask*input_array_2d,axis=0,keepdims=1)/mask_sum
    array_nomean_2d = input_array_2d - array_mean_2d
    array_nomean_2d[~mask] = 0
    
    good_ind_counts = np.unique(mask_sum)
    good_ind_n_at_counts = np.empty((0,)).astype('int64')
    for good_ind_count in good_ind_counts:
        good_ind_n_at_counts = np.hstack((good_ind_n_at_counts,int(np.sum(mask_sum == good_ind_count))))
    
    if len(seasonal_output_times) > 0:
        G_seasonal_times = np.ones((seasonal_output_tvalues.size,1))
        for num_har in range(1,num_harmonics+1):
            G_seasonal_times = np.hstack((G_seasonal_times,np.cos(((2*np.pi*num_har)/365.24)*seasonal_output_tvalues),\
                                                           np.sin(((2*np.pi*num_har)/365.24)*seasonal_output_tvalues)))        
        seasonal_cycle_2d = np.empty((seasonal_output_tvalues.size,array_nomean_2d.shape[-1]))
        seasonal_cycle_2d.fill(np.nan)
    
    array_noseason_2d = np.empty(array_nomean_2d.shape)
    array_noseason_2d.fill(np.nan)
    for good_ind_count,good_ind_n_at_count in zip(good_ind_counts,good_ind_n_at_counts):
        if good_ind_count < 0.5:
            continue
        curr_at_count_ind = (mask_sum == good_ind_count).nonzero()[-1]
        while len(curr_at_count_ind) > 0:
            curr_ind = curr_at_count_ind[0]
            curr_good = (mask[:,curr_ind] == True).nonzero()[0]
            same_good_ind = (np.abs(np.sum(mask[:,[curr_ind]]*mask[:,curr_at_count_ind],axis=0)\
                                    - np.sum(mask[:,[curr_ind]])) < 1.e-5).nonzero()[-1]
            same_good_ind = curr_at_count_ind[same_good_ind]
                        
            G = np.ones((curr_good.size,1))
            
            # ensure that number of basis functions is not greater than data points
            curr_num_harmonics = np.fmin(int(np.ceil(len(curr_good)/2)-1),num_harmonics)
            for num_har in range(1,curr_num_harmonics+1):
                G = np.hstack((G,np.cos(((2*np.pi*num_har)/365.24)*time_values[curr_good]),\
                               np.sin(((2*np.pi*num_har)/365.24)*time_values[curr_good])))
            G_min_std = np.nanmin(np.std(G[:,1:],axis=0))
            if G_min_std < 2**(-3/2):
                # not enough diversity of data to fit harmonics; just fit mean
                print('Not enough diversity in sampling points to fit harmonics')
                print('Fitting mean only')
                curr_num_harmonics = 0
                G = G[:,[0]]
            
            ind_tuple = (np.expand_dims(curr_good,axis=-1),same_good_ind)
            m = np.dot(np.linalg.inv(np.matmul(G.transpose(),G)),\
                       np.dot(G.transpose(),array_nomean_2d[ind_tuple]))
            array_noseason_2d[ind_tuple] = array_nomean_2d[ind_tuple]\
                                                               - np.dot(G,m)
            
            if len(seasonal_output_times) > 0:
                seasonal_cycle_2d[:,same_good_ind] = np.dot(G_seasonal_times[:,:m.shape[0]],m)
                seasonal_cycle_2d += array_mean_2d
            
            curr_at_count_ind = np.setdiff1d(curr_at_count_ind,same_good_ind,assume_unique=True)
    
    array_noseason = np.moveaxis(np.reshape(array_noseason_2d,input_array_moveaxis.shape),0,time_axis_num)
    
    
    if len(seasonal_output_times) > 0:
        seasonal_cycle = np.moveaxis(np.reshape(seasonal_cycle_2d,
                                            (seasonal_cycle_2d.shape[0],) + input_array_moveaxis.shape[1:])\
                                            ,0,time_axis_num)
        return array_noseason,seasonal_cycle
    else:
        return array_noseason



#===============================================================================


def trend_compute(input_array,axis=-1,delta_dim=1,output_trendline=False):
    """
    Computes linear regression-based trend in input_array, along specified axis with interval delta_dim.
    Outputs trend values in reg_trend.
    If output_trendline = True, also outputs trend lines (that can be subtracted from input_array) in
    reg_trendline."""
    
    pass
    
    import numpy as np
    
    input_reshaped = np.moveaxis(input_array,axis,0)
    input_shape = input_reshaped.shape
    dim_vec = np.arange(0.,input_shape[0])
    mask = np.asarray(np.logical_and(np.logical_and(~np.isnan(input_reshaped),~np.isinf(input_reshaped)),\
                                      np.abs(input_reshaped) > 1e-10*np.nanmax(np.abs(input_reshaped))))
    dim_vec_tiled = np.moveaxis(np.tile(dim_vec,np.concatenate((input_shape[1:],[1]))),-1,0)
    dim_vec_nomean = dim_vec_tiled - (np.sum(mask*dim_vec_tiled,axis=0)/np.sum(mask,axis=0))
    input_mean = np.nansum(mask*input_reshaped,axis=0,keepdims=True)/np.nansum(mask,axis=0,keepdims=True)
    input_nomean = input_reshaped - input_mean
    reg_num = np.nansum(mask*(dim_vec_nomean*input_nomean),axis=0,keepdims=True)
    reg_denom = np.nansum(mask*(dim_vec_nomean**2),axis=0,keepdims=True)
    reg_trend = (np.moveaxis(reg_num/reg_denom,0,axis))/delta_dim
    if output_trendline == True:
        reg_trendline = np.moveaxis(input_mean,0,axis) + (delta_dim*reg_trend*np.moveaxis(dim_vec_nomean,0,axis))
        return reg_trend,reg_trendline
    else:    
        return reg_trend



#===============================================================================


def interp1_fft(interp_coords,base_coords,base_values,interp_axis_num=0,filter_cutoff=0.8,output_trendline=False):
    """Interpolate n-dimensional array base_values along interp_axis_num, from base_coords to interp_coords.
    The interpolation includes a low-pass fft-based filter with a half-power cutoff at a wavelength of
    filter_cutoff*(2*interp_spacing), where 2*interp_spacing is the approximate Nyquist wavelength
    of interp_coords.
    Outputs the interpolated values, and (if output_trendline=True) 
    the mean plus linear trend along the interpolation axis."""
    
    pass
    
    import numpy as np
    from scipy import special
    
    interp_c = np.reshape(np.interp(interp_coords,base_coords,np.arange(len(base_coords))),(-1,1))
    base_values_moveaxis = np.moveaxis(base_values,interp_axis_num,0)
    base_values_2d = np.reshape(base_values_moveaxis,(base_values.shape[interp_axis_num],-1))
    mask = np.logical_and(np.logical_and(~np.isnan(base_values_2d),~np.isinf(base_values_2d)),\
                          np.abs(base_values_2d) > 1.e-15*np.nanmax(np.abs(base_values_2d)))
    base_c_2d = np.tile(np.reshape(np.arange(len(base_coords)),(-1,1)),(1,base_values_2d.shape[1]))
    base_c_mean = np.nansum(mask*base_c_2d,axis=0,keepdims=1)/(np.nansum(mask,axis=0,keepdims=1))
    base_c_nomean = base_c_2d - base_c_mean
    base_values_mean = np.nansum(mask*base_values_2d,axis=0,keepdims=1)/\
                                (np.nansum(mask,axis=0,keepdims=1))
    base_values_nomean = base_values_2d - base_values_mean
    base_values_trendval = np.nansum(mask*base_c_nomean*base_values_nomean,axis=0)/\
                                (np.nansum(mask*(base_c_nomean**2),axis=0))
    base_values_trend = base_c_nomean*base_values_trendval
    base_values_notrend = base_values_2d - base_values_mean - base_values_trend
    base_values_notrend[~mask] = 0
    fft_coeffs = (np.fft.fft(base_values_notrend,axis=0))/len(base_coords)
    
    f_vec = ((np.arange(len(base_coords))/len(base_coords) + (1/2)) % 1) - (1/2)
    mean_ratio_interp = np.nanmean(np.diff(interp_c,axis=0))
    high_bound = 1/(filter_cutoff*2*mean_ratio_interp)
    
    from scipy import special
    steepness_factor = 2
    half_power_adj = np.exp(special.erfinv((2**(1/2)) - 1)/steepness_factor)
    bandpass_filter = 0.5 - (0.5*(special.erf(steepness_factor*(np.log(np.abs(f_vec))\
                                                                - np.log(high_bound*half_power_adj)))))
    fft_coeffs = np.expand_dims(bandpass_filter,axis=1)*fft_coeffs
    
    output_mean_ptrend = ((interp_c - base_c_mean)*base_values_trendval) + base_values_mean
    output_values_2d = (output_mean_ptrend.copy()).astype('complex128')
    for f_v,fft_cf in zip(f_vec,fft_coeffs):
        output_values_2d += fft_cf*np.tile(np.exp(1j*(2*np.pi*f_v)*interp_c),(1,base_values_2d.shape[1]))
    
    output_values_2d = np.real(output_values_2d)
    output_moveaxis_shape = (len(interp_coords),) + base_values_moveaxis.shape[1:]
    output_mean_ptrend = np.moveaxis(np.reshape(output_mean_ptrend,output_moveaxis_shape),0,interp_axis_num)
    output_values = np.moveaxis(np.reshape(output_values_2d,output_moveaxis_shape),0,interp_axis_num)
    
    if output_trendline == True:
        return output_values,output_mean_ptrend
    else:
        return output_values



#===============================================================================


def interp1_ndarray(interp_coords,input_coords,input_array,axis=-1):
    """1-D linear interpolation along specified axis, using 1-D or N-D monotonically-increasing input_coords
    and N-D input_array as a basis, and interpolating to 1-D or N-D array interp_coords along specified axis."""
    
    pass
    
    import numpy as np
    
    input_array_origshape = input_array.shape
    
    interp_coords = np.moveaxis(interp_coords,axis,0)
    input_coords = np.moveaxis(input_coords,axis,0)
    input_array = np.moveaxis(input_array,axis,0)
    
    if input_coords.size == np.nanmax(np.asarray(input_coords.shape)):
        input_coords = np.reshape(input_coords,(-1,1))
    if input_array.size == np.nanmax(np.asarray(input_array.shape)):
        input_array = np.reshape(input_array,(-1,1))
    if interp_coords.size == np.nanmax(np.asarray(interp_coords.shape)):
        interp_coords = np.reshape(interp_coords,(-1,1))
    
    interp_array_reshaped = np.empty((interp_coords.shape[0],)+input_array.shape[1:])
    interp_array_reshaped.fill(np.nan)
    for curr_interp_ind in range(interp_coords.shape[0]):
        curr_interp_coord = interp_coords[curr_interp_ind,:]
        diff_coords = curr_interp_coord - input_coords
        curr_input_bins = np.unique(np.logical_and(diff_coords[:-1,:] >= 0,\
                                                 diff_coords[1:,:] <= 0).nonzero()[0])
        for input_bin in curr_input_bins:
            in_input_bin = np.logical_and(diff_coords[input_bin,:] >= 0,\
                                         diff_coords[input_bin+1,:] <= 0).nonzero()
            
            if input_coords.size == input_coords.shape[0]:
                input_coord_lower = input_coords[input_bin]
                input_coord_higher = input_coords[input_bin+1]
                input_array_lower = input_array[input_bin,:].flatten()
                input_array_higher = input_array[input_bin+1,:].flatten()
            else:
                input_ind_lower = ((input_bin*np.ones((len(in_input_bin[0]),)).astype('int64')),)+in_input_bin
                input_ind_higher = (((input_bin+1)*np.ones((len(in_input_bin[0]),)).astype('int64')),)+in_input_bin
                input_coord_lower = input_coords[input_ind_lower]
                input_coord_higher = input_coords[input_ind_higher]
                input_array_lower = input_array[input_ind_lower]
                input_array_higher = input_array[input_ind_higher]
            if interp_coords.size != interp_coords.shape[0]:
                curr_interp_coord_inbin = curr_interp_coord[in_input_bin]
            elif input_coords.size != input_coords.shape[0]:
                curr_interp_coord_inbin = curr_interp_coord.flatten()
            else:
                curr_interp_coord_inbin = curr_interp_coord
            
            weight_lower = (input_coord_higher - curr_interp_coord_inbin)/(input_coord_higher - input_coord_lower)
            curr_interp_array = (weight_lower*input_array_lower) + ((1 - weight_lower)*input_array_higher)
            if ((input_coords.size == input_coords.shape[0]) and (interp_coords.size == interp_coords.shape[0])):
                interp_array_reshaped[curr_interp_ind,:] = np.reshape(curr_interp_array,input_array.shape[1:])
            else:
                output_ind = ((curr_interp_ind*np.ones((len(in_input_bin[0]),)).astype('int64')),)+in_input_bin
                interp_array_reshaped[output_ind] = curr_interp_array
    
    interp_array = np.moveaxis(interp_array_reshaped,0,axis)
    if len(input_array_origshape) < 2:
        interp_array = interp_array.flatten()
    
    return interp_array