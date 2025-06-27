# statistical functions

def correlation_scalar_scalar_uncert(input_array_1,input_array_2,axis=-1,delta_dim=1,delta_lag=1,lag_range_to_test=(0,10),confidence_level=0.95):
    """Compute lagged correlation of input_array_1 and input_array_2.
    
    This function computes the lagged correlation of input_array_1 and input_array_2 along specified number axis.
    Other input parameters include
    delta_dim: separation between adjacent coordinates along specified axis
    delta_lag: separation between lag values to test correlation
    lag_range_to_test: 2-element tuple indicating range to compute lags over
    confidence_level: confidence level for (two-tailed) uncertainty bounds, expressed as a value between 0 and 1.
    
    Outputs included in list_of_outputs (all numpy arrays):
    [0]: correlation_array_at_lags, correlation coefficients at lags specified in list_of_outputs[6]
    [1]: correlation_dof_array_zero_lag, degrees of freedom in zero-lag correlation
    [2]: integral_scale_1, scale (in units of axis axis) over which input_array_1 data can be considered approx. independent
    [3]: integral_scale_2, scale (in units of axis axis) over which input_array_2 data can be considered approx. independent
    [4]: correlation_array_low_mag_bound, lower-magnitude confidence bound of correlation coefficients based on confidence_level specified
    [5]: correlation_array_high_mag_bound, higher-magnitude confidence bound of correlation coefficients based on confidence_level specified
    [6]: lags, array of lags over which correlation was computed"""
    pass
    
    import numpy as np
    from scipy import special
    
    
    # reshape input arrays if needed
    
    if input_array_1.shape != input_array_2.shape:
        input_array_1 = input_array_1*np.ones(input_array_2.shape)
        input_array_2 = input_array_2*np.ones(input_array_1.shape)
    
    orig_array_shape = input_array_1.shape
    if len(orig_array_shape) == 1:
        input_array_1 = np.reshape(input_array_1,(1,-1))
        input_array_2 = np.reshape(input_array_2,(1,-1))
        if axis == 0:
             axis = -1
    
    
    # permute and reshape arrays to standard shape
    dim_length = input_array_1.shape[axis]
    size_array = np.array(input_array_1.shape)
    prod_size_not_inc_corr_dim = np.prod(size_array[np.delete(np.arange(len(size_array)),axis)])
    input_array_1_permuted = np.reshape(np.moveaxis(input_array_1,axis,-1),(prod_size_not_inc_corr_dim,dim_length))
    input_array_2_permuted = np.reshape(np.moveaxis(input_array_2,axis,-1),(prod_size_not_inc_corr_dim,dim_length))
    
    input_array_1_masked = np.ma.array(input_array_1_permuted,mask=np.asarray(np.logical_or(np.logical_or(np.isnan(input_array_1_permuted),np.isinf(input_array_1_permuted)),np.abs(input_array_1_permuted) < (1.e-15)*np.nanmax(np.abs(input_array_1_permuted[~np.isinf(input_array_1_permuted)])))))
    input_array_2_masked = np.ma.array(input_array_2_permuted,mask=np.asarray(np.logical_or(np.logical_or(np.isnan(input_array_2_permuted),np.isinf(input_array_2_permuted)),np.abs(input_array_2_permuted) < (1.e-15)*np.nanmax(np.abs(input_array_2_permuted[~np.isinf(input_array_2_permuted)])))))
    
    # remove means and trends
    index_array_tiled = np.tile(np.r_[0:dim_length],(prod_size_not_inc_corr_dim,1))
    index_array_nomean = np.ma.array(index_array_tiled - np.tile((np.ma.array(index_array_tiled,mask=(input_array_1_masked).mask)).mean(axis=-1,keepdims=1),(1,dim_length)),mask=input_array_1_masked.mask)
    input_array_1_nomean = input_array_1_permuted - np.tile(np.ma.mean(input_array_1_masked,axis=-1,keepdims=1),(1,dim_length))
    input_array_1_detrend = input_array_1_nomean - (index_array_nomean*np.tile((1/((index_array_nomean**2).sum(axis=-1,keepdims=1)))*((index_array_nomean*input_array_1_nomean).sum(axis=-1,keepdims=1)),(1,dim_length)))
    input_array_1_masked = np.ma.array(input_array_1_detrend,mask=(input_array_1_masked).mask)
    index_array_nomean = np.ma.array(index_array_tiled - np.tile((np.ma.array(index_array_tiled,mask=(input_array_2_masked).mask)).mean(axis=-1,keepdims=1),(1,dim_length)),mask=input_array_2_masked.mask)
    input_array_2_nomean = input_array_2_permuted - np.tile(np.ma.mean(input_array_2_masked,axis=-1,keepdims=1),(1,dim_length))
    input_array_2_detrend = input_array_2_nomean - (index_array_nomean*np.tile((1/((index_array_nomean**2).sum(axis=-1,keepdims=1)))*((index_array_nomean*input_array_2_nomean).sum(axis=-1,keepdims=1)),(1,dim_length)))
    input_array_2_masked = np.ma.array(input_array_2_detrend,mask=(input_array_2_masked).mask)
    
    # compute integral scales & degrees of freedom
    delta_lag_ind = np.round(delta_lag/delta_dim)
    lags = delta_dim*np.round((delta_lag*np.round((np.r_[np.ceil(lag_range_to_test[0]/delta_lag):(np.floor(lag_range_to_test[1]/delta_lag) + 0.5)])))/delta_dim)
    if np.nanmax(np.abs(lags)) >= (delta_lag_ind*dim_length):
        lags = lags[np.abs(lags) < (delta_lag_ind*delta_dim*(dim_length/2))]
    autocorr_lags = (delta_dim*delta_lag_ind)*(np.r_[1:((np.nanmax(np.abs(lags))/(delta_dim*delta_lag_ind)) + 0.5)])
    
    N = len(autocorr_lags)
    
    lag_step = (delta_lag_ind).astype('i8')   # number of time points between lags to test for dof
    
    input_1_autocorr_at_lags = np.empty((prod_size_not_inc_corr_dim,N))
    input_1_autocorr_at_lags.fill(np.nan)
    input_2_autocorr_at_lags = np.empty((prod_size_not_inc_corr_dim,N))
    input_2_autocorr_at_lags.fill(np.nan)
    input_1_integral_scale = np.empty((prod_size_not_inc_corr_dim,1))
    input_1_integral_scale.fill(np.nan)
    input_2_integral_scale = np.empty((prod_size_not_inc_corr_dim,1))
    input_2_integral_scale.fill(np.nan)
    
    n_lag_step = 0
    lag_ind = lag_step
    complete_flag = 0
    while ((complete_flag == 0) and (n_lag_step < N)):
        part_1_ind = (np.r_[0:(dim_length - lag_ind)]).astype('i8')
        part_2_ind = (np.r_[lag_ind:dim_length]).astype('i8')
    
        mask_combined = np.logical_or((input_array_1_masked[:,part_1_ind]).mask,(input_array_1_masked[:,part_2_ind]).mask)
        part_1_mean = (np.ma.array(input_array_1_masked[:,part_1_ind],mask=mask_combined)).mean(axis=-1,keepdims=1)
        part_2_mean = (np.ma.array(input_array_1_masked[:,part_2_ind],mask=mask_combined)).mean(axis=-1,keepdims=1)
        part_1_corr = np.ma.array(input_array_1_masked[:,part_1_ind] - np.tile(part_1_mean,(1,dim_length - lag_ind)),mask=mask_combined)
        part_2_corr = np.ma.array(input_array_1_masked[:,part_2_ind] - np.tile(part_2_mean,(1,dim_length - lag_ind)),mask=mask_combined)
        input_1_autocorr_at_lags[:,n_lag_step] = (np.ma.sum(part_1_corr*part_2_corr,axis=-1))/(((np.ma.sum(part_1_corr**2,axis=-1))**(1/2))*((np.ma.sum(part_2_corr**2,axis=-1))**(1/2)))
        input_1_autocorr_at_lags[np.sum(~mask_combined,axis=-1) < 0.5,n_lag_step] = np.nan
        mask_combined = np.logical_or((input_array_2_masked[:,part_1_ind]).mask,(input_array_2_masked[:,part_2_ind]).mask)
        part_1_mean = (np.ma.array(input_array_2_masked[:,part_1_ind],mask=mask_combined)).mean(axis=-1,keepdims=1)
        part_2_mean = (np.ma.array(input_array_2_masked[:,part_2_ind],mask=mask_combined)).mean(axis=-1,keepdims=1)
        part_1_corr = np.ma.array(input_array_2_masked[:,part_1_ind] - np.tile(part_1_mean,(1,dim_length - lag_ind)),mask=mask_combined)
        part_2_corr = np.ma.array(input_array_2_masked[:,part_2_ind] - np.tile(part_2_mean,(1,dim_length - lag_ind)),mask=mask_combined)
        input_2_autocorr_at_lags[:,n_lag_step] = (np.ma.sum(part_1_corr*part_2_corr,axis=-1))/(((np.ma.sum(part_1_corr**2,axis=-1))**(1/2))*((np.ma.sum(part_2_corr**2,axis=-1))**(1/2)))
        input_2_autocorr_at_lags[np.sum(~mask_combined,axis=-1) < 0.5,n_lag_step] = np.nan
        if n_lag_step == N - 1:
            first_neg_1 = np.logical_and(np.reshape(np.isnan(input_1_integral_scale),(prod_size_not_inc_corr_dim,1)),np.reshape(~np.isnan(input_1_autocorr_at_lags[:,0]),(prod_size_not_inc_corr_dim,1)))
            first_neg_2 = np.logical_and(np.reshape(np.isnan(input_2_integral_scale),(prod_size_not_inc_corr_dim,1)),np.reshape(~np.isnan(input_2_autocorr_at_lags[:,0]),(prod_size_not_inc_corr_dim,1)))
        else:
            first_neg_1 = np.logical_and(np.reshape(np.logical_or(input_1_autocorr_at_lags[:,n_lag_step] < 0,np.logical_and(np.isnan(input_1_autocorr_at_lags[:,n_lag_step]),~np.isnan(input_1_autocorr_at_lags[:,0]))),(prod_size_not_inc_corr_dim,1)),np.reshape(np.isnan(input_1_integral_scale),(prod_size_not_inc_corr_dim,1)))
            first_neg_2 = np.logical_and(np.reshape(np.logical_or(input_2_autocorr_at_lags[:,n_lag_step] < 0,np.logical_and(np.isnan(input_2_autocorr_at_lags[:,n_lag_step]),~np.isnan(input_2_autocorr_at_lags[:,0]))),(prod_size_not_inc_corr_dim,1)),np.reshape(np.isnan(input_2_integral_scale),(prod_size_not_inc_corr_dim,1)))
        first_neg_mask = np.tile(first_neg_1,(1,n_lag_step + 1))
        input_1_autocorr_at_lags_masked = np.ma.array(input_1_autocorr_at_lags,mask=np.isnan(input_1_autocorr_at_lags))
        curr_input_1_integral_scale = delta_dim*lag_step*(np.reshape(np.ceil(first_neg_mask[:,0]),(prod_size_not_inc_corr_dim,1)) + np.fmax(np.zeros((prod_size_not_inc_corr_dim,1)),(2*np.ma.sum(first_neg_mask*input_1_autocorr_at_lags_masked[:,0:(n_lag_step+1)],axis=1,keepdims=1))))
        if n_lag_step > 0:
            nan_case_integral_scale = np.fmin(delta_dim*lag_ind*np.ones((prod_size_not_inc_corr_dim,1)),(delta_dim*lag_step*(np.reshape(first_neg_mask[:,0],(-1,1)) + (2*np.sum(first_neg_mask[:,0:n_lag_step]*input_1_autocorr_at_lags[:,0:n_lag_step],axis=1,keepdims=1)))))
            curr_input_1_integral_scale[np.logical_and(first_neg_1,np.isnan(curr_input_1_integral_scale))] = nan_case_integral_scale[np.logical_and(first_neg_1,np.isnan(curr_input_1_integral_scale))]
        input_1_integral_scale[first_neg_1] = curr_input_1_integral_scale[first_neg_1]
        first_neg_mask = np.tile(first_neg_2,(1,n_lag_step + 1))
        input_2_autocorr_at_lags_masked = np.ma.array(input_2_autocorr_at_lags,mask=np.isnan(input_2_autocorr_at_lags))
        curr_input_2_integral_scale = delta_dim*lag_step*(np.reshape(np.ceil(first_neg_mask[:,0]),(prod_size_not_inc_corr_dim,1)) + np.fmax(np.zeros((prod_size_not_inc_corr_dim,1)),(2*np.ma.sum(first_neg_mask*input_2_autocorr_at_lags_masked[:,0:(n_lag_step+1)],axis=1,keepdims=1))))
        if n_lag_step > 0:
            nan_case_integral_scale = np.fmin(delta_dim*lag_ind*np.ones((prod_size_not_inc_corr_dim,1)),(delta_dim*lag_step*(np.reshape(first_neg_mask[:,0],(-1,1)) + (2*np.sum(first_neg_mask[:,0:n_lag_step]*input_2_autocorr_at_lags[:,0:n_lag_step],axis=1,keepdims=1)))))
            curr_input_2_integral_scale[np.logical_and(first_neg_2,np.isnan(curr_input_2_integral_scale))] = nan_case_integral_scale[np.logical_and(first_neg_2,np.isnan(curr_input_2_integral_scale))]
        input_2_integral_scale[first_neg_2] = curr_input_2_integral_scale[first_neg_2]
    
        if ~np.logical_or(np.sum(np.isnan(input_1_integral_scale)),np.sum(np.isnan(input_2_integral_scale))):
            complete_flag = 1
        n_lag_step = n_lag_step + 1
        lag_ind = lag_ind + lag_step
    input_corr_integral_scale = np.fmin(input_1_integral_scale,input_2_integral_scale)
    corr_dof_array = (delta_dim*np.sum(np.logical_and(~input_array_1_masked.mask,~input_array_2_masked.mask),axis=1,keepdims=1))/input_corr_integral_scale
    
    integral_scale_1 = np.moveaxis(np.reshape(input_1_integral_scale,np.concatenate((size_array[np.delete(np.arange(len(size_array)),axis)],np.array([1])))),-1,axis)
    integral_scale_2 = np.moveaxis(np.reshape(input_2_integral_scale,np.concatenate((size_array[np.delete(np.arange(len(size_array)),axis)],np.array([1])))),-1,axis)
    correlation_dof_array_zero_lag = np.moveaxis(np.reshape(corr_dof_array,np.concatenate((size_array[np.delete(np.arange(len(size_array)),axis)],np.array([1])))),-1,axis)
    
    corrcoeff_array = np.empty((prod_size_not_inc_corr_dim,len(lags)))
    corrcoeff_array.fill(np.nan)
    corrcoeff_low_mag = np.empty((prod_size_not_inc_corr_dim,len(lags)))
    corrcoeff_low_mag.fill(np.nan)    
    corrcoeff_high_mag = np.empty((prod_size_not_inc_corr_dim,len(lags)))
    corrcoeff_high_mag.fill(np.nan)
    for n_lag_ind,curr_lag in enumerate(lags):
        curr_lag = lags[n_lag_ind]
        lag_ind = (np.round(curr_lag/delta_dim)).astype('i8')
        input_1_corr_ind = (np.r_[(np.fmax(0,-lag_ind)):(np.fmin(dim_length,dim_length - lag_ind))]).astype('i8')
        input_2_corr_ind = (np.r_[(np.fmax(0,lag_ind)):(np.fmin(dim_length,dim_length + lag_ind))]).astype('i8')
    
        mask_combined = np.logical_or((input_array_1_masked[:,input_1_corr_ind]).mask,(input_array_2_masked[:,input_2_corr_ind]).mask)
        part_1_mean = (np.ma.array(input_array_1_masked[:,input_1_corr_ind],mask=mask_combined)).mean(axis=-1,keepdims=1)
        part_2_mean = (np.ma.array(input_array_2_masked[:,input_2_corr_ind],mask=mask_combined)).mean(axis=-1,keepdims=1)
        part_1_corr = np.ma.array(input_array_1_masked[:,input_1_corr_ind] - np.tile(part_1_mean,(1,dim_length - np.abs(lag_ind))),mask=mask_combined)
        part_2_corr = np.ma.array(input_array_2_masked[:,input_2_corr_ind] - np.tile(part_2_mean,(1,dim_length - np.abs(lag_ind))),mask=mask_combined)
        corrcoeff_array[:,n_lag_ind] = (np.ma.sum(part_1_corr*part_2_corr,axis=-1))/(((np.ma.sum(part_1_corr**2,axis=-1))**(1/2))*((np.ma.sum(part_2_corr**2,axis=-1))**(1/2)))
        corrcoeff_array[np.sum(~mask_combined,axis=-1) < 0.5,n_lag_ind] = np.nan
    
        curr_corr_dof_array = (delta_dim*np.sum(~mask_combined,axis=-1,keepdims=1))/input_corr_integral_scale
    
        stderror_Z = (curr_corr_dof_array - 3)**(-1/2)
        stderror_Z[curr_corr_dof_array < 3] = np.inf
    
        normalized_Z_value = 0.5*(np.log(1 + np.abs(corrcoeff_array[:,n_lag_ind])) - np.log(1 - np.abs(corrcoeff_array[:,n_lag_ind])))
        Z_confidence_lower_mag = normalized_Z_value - ((stderror_Z.flatten())*(special.erfinv(confidence_level))*(2**(1/2)))
        r_confidence_lower_mag = 1 - (2/(np.exp(2*Z_confidence_lower_mag) + 1))
        Z_confidence_higher_mag = normalized_Z_value + ((stderror_Z.flatten())*(special.erfinv(confidence_level))*(2**(1/2)))
        r_confidence_higher_mag = 1 - (2/(np.exp(2*Z_confidence_higher_mag) + 1))
    
        corrcoeff_low_mag[:,n_lag_ind] = (np.sign(corrcoeff_array[:,n_lag_ind]))*r_confidence_lower_mag
        corrcoeff_high_mag[:,n_lag_ind] = (np.sign(corrcoeff_array[:,n_lag_ind]))*r_confidence_higher_mag
    correlation_array_at_lags = np.moveaxis(np.reshape(corrcoeff_array,np.concatenate((size_array[np.delete(np.arange(len(size_array)),axis)],np.array([len(lags)])))),-1,axis)
    correlation_array_low_mag_bound = np.moveaxis(np.reshape(corrcoeff_low_mag,np.concatenate((size_array[np.delete(np.arange(len(size_array)),axis)],np.array([len(lags)])))),-1,axis)
    correlation_array_high_mag_bound = np.moveaxis(np.reshape(corrcoeff_high_mag,np.concatenate((size_array[np.delete(np.arange(len(size_array)),axis)],np.array([len(lags)])))),-1,axis)
    
    
    # reshape output arrays to 1-D if needed to match input array size
    if len(orig_array_shape) == 1:
        correlation_array_at_lags = np.reshape(correlation_array_at_lags,(-1,))
        correlation_dof_array_zero_lag = np.reshape(correlation_dof_array_zero_lag,(1,))
        integral_scale_1 = np.reshape(integral_scale_1,(-1,))
        integral_scale_2 = np.reshape(integral_scale_2,(-1,))
        correlation_array_low_mag_bound = np.reshape(correlation_array_low_mag_bound,(-1,))
        correlation_array_high_mag_bound = np.reshape(correlation_array_high_mag_bound,(-1,))
        lags = np.reshape(lags,(-1,))
    
    
    list_of_outputs = [correlation_array_at_lags,correlation_dof_array_zero_lag,integral_scale_1,integral_scale_2,correlation_array_low_mag_bound,correlation_array_high_mag_bound,lags]
    return list_of_outputs
