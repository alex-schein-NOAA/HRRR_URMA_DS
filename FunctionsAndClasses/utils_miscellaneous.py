from FunctionsAndClasses.HEADER_utilities import *

########################################################
### MISCELLANEOUS FUNCTIONS
# Any functions that do anything else
########################################################

def rolling_avg(x, window_len):
    return np.convolve(x, np.ones(window_len), 'valid') / window_len

########################################################

def extract_numbers(text):
    pattern = r'\d+\.\d+|\.\d+|\d+'
    return re.findall(pattern, text) #returns a LIST of all number (including decimal) STRINGS in the original 'text' string, so make sure to subselect and cast appropriately!

########################################################

def read_epoch_num_and_loss(TRAINING_LOG_FILEPATH):
    """
    Input: full filepath to the training log whose epoch number and losses we want
        !!! LINES MUST BE FORMATTED AS SOMETHING LIKE "End of epoch [number] | Average epoch loss = [float] | [whatever]"

    Output: Two arrays, the first containing the epoch numbers and the second containing the corresponding loss values
    """

    epoch_number_arr = []
    epoch_loss_arr = []
    
    with open(TRAINING_LOG_FILEPATH, 'r') as file:
        for line_number, line in enumerate(file):
                if line.startswith('End of epoch'): #Filters out any lines we don't care about - but the file MUST be formatted correctly!
                    split_line = line.split('|')
                    epoch_number = int(extract_numbers(split_line[0])[0])
                    epoch_number_arr.append(epoch_number)
    
                    epoch_loss = float(extract_numbers(split_line[1])[0])
                    epoch_loss_arr.append(epoch_loss)

    return epoch_number_arr, epoch_loss_arr

########################################################

def calc_gradient_difference(input_arr, target_arr, is_zonal):
    """ 
    Calculates the difference in the gradients of input_arr (can be model output or smartinit) and target_arr (i.e. URMA). 
    Arrays should be spatially subset to the region of interest before being fed into this function.
    NOTE: uses grid spacing = 2.5, which is hardcoded! All data here is on a 2.5km grid so this should be fine, though

    Inputs:
        - input_arr --> already-restricted array of input data (i.e. model output or Smartinit)
        - target_arr --> already-restricted array of target data (i.e. URMA)
        - is_zonal --> bool; if true, the zonal (x) gradient difference is returned, and if False then the meridional (y) gradient difference is returned
    """
    if is_zonal:
        return (np.gradient(input_arr, 2.5)[1] - np.gradient(target_arr, 2.5)[1])
    else:
        return (np.gradient(input_arr, 2.5)[0] - np.gradient(target_arr, 2.5)[0])

########################################################

def calc_RMSE(input_arr, target_arr=None):
    """
    Simple overloaded function to calculate RMSE from either one or two inputs. If only input_arr is given (i.e. an array of differences) then just returns np.sqrt(np.nanmean(input_arr**2)). If target_arr is also given then it calculates np.sqrt(np.nanmean((input_arr - target_arr)**2))
    """
    if target_arr is None:
        return np.sqrt(np.nanmean(input_arr**2))
    else:
        return np.sqrt(np.nanmean((input_arr-target_arr)**2))

########################################################

def calc_fourier_RMSE(input_arr, target_arr):
    """ 
    Function to calculate Fourier RMSE between 2 arrays. Note that the resulting RMSE is taken between absolutes, that is, the power spectrum of both input_arr and target_arr are taken, then their difference is taken, rather than taking the difference beforehand (i.e. involving the complex part).
    Inputs should be 2D arrays, already spatially retricted to the region of interest.
    """

    ps_input = np.absolute(np.fft.fft2(input_arr))
    ps_target = np.absolute(np.fft.fft2(target_arr))

    return calc_RMSE(ps_input, ps_target)

########################################################

def calc_all_stats_all_regions(model_attrs, TARG_VAR):
    """
    Calculates all stats (domain-average RMSE, gradient RMSEs [zonal, meridional, pythagorean], and Fourier RMSE) for a given model across all geographical subdomains.
    Should only be called once, to write data to disk. Should NOT be used to read data from disk - use check_rmse_data_and_read_in for that
    This function, as written, is an abomination and desparately needs some heavy reformatting, but it works for now and likely won't get that much use.
    
    Inputs:
        - model_attrs --> instance of DefineModelAttributes with .dataset already initialized and model weights loaded.
        - TARG_VAR --> string defining the target variable (e.g. 't2m'). Note this will also technically be used as the predictor variable, but model initialization is handled outside this function so multipredictor models should be fine.
    """
    
    C = CONSTANTS()
    
    REGION_KEYWORD_LIST = list(C.region_keyword_dict.keys()) #for convenience
    
    model_domain_avg_rmse_regional_statsobjs_list = [StatsObject(is_smartinit=False, is_conus=False, region_keyword=x, current_model_attrs=model_attrs, predictor_var=TARG_VAR, target_var=TARG_VAR) for x in REGION_KEYWORD_LIST]
    model_zonal_gradient_rmse_regional_statsobjs_list = [StatsObject(is_smartinit=False, is_conus=False, region_keyword=x, current_model_attrs=model_attrs, predictor_var=TARG_VAR, target_var=TARG_VAR) for x in REGION_KEYWORD_LIST]
    model_merid_gradient_rmse_regional_statsobjs_list = [StatsObject(is_smartinit=False, is_conus=False, region_keyword=x, current_model_attrs=model_attrs, predictor_var=TARG_VAR, target_var=TARG_VAR) for x in REGION_KEYWORD_LIST]
    model_pythag_gradient_rmse_regional_statsobjs_list = [StatsObject(is_smartinit=False, is_conus=False, region_keyword=x, current_model_attrs=model_attrs, predictor_var=TARG_VAR, target_var=TARG_VAR) for x in REGION_KEYWORD_LIST]
    model_fourier_rmse_regional_statsobjs_list = [StatsObject(is_smartinit=False, is_conus=False, region_keyword=x, current_model_attrs=model_attrs, predictor_var=TARG_VAR, target_var=TARG_VAR) for x in REGION_KEYWORD_LIST]
    
    smartinit_domain_avg_rmse_regional_statsobjs_list = [StatsObject(is_smartinit=True, is_conus=False, region_keyword=x, predictor_var=TARG_VAR, target_var=TARG_VAR) for x in REGION_KEYWORD_LIST]
    smartinit_zonal_gradient_rmse_regional_statsobjs_list = [StatsObject(is_smartinit=True, is_conus=False, region_keyword=x, predictor_var=TARG_VAR, target_var=TARG_VAR) for x in REGION_KEYWORD_LIST]
    smartinit_merid_gradient_rmse_regional_statsobjs_list = [StatsObject(is_smartinit=True, is_conus=False, region_keyword=x, predictor_var=TARG_VAR, target_var=TARG_VAR) for x in REGION_KEYWORD_LIST]
    smartinit_pythag_gradient_rmse_regional_statsobjs_list = [StatsObject(is_smartinit=True, is_conus=False, region_keyword=x, predictor_var=TARG_VAR, target_var=TARG_VAR) for x in REGION_KEYWORD_LIST]
    smartinit_fourier_rmse_regional_statsobjs_list = [StatsObject(is_smartinit=True, is_conus=False, region_keyword=x, predictor_var=TARG_VAR, target_var=TARG_VAR) for x in REGION_KEYWORD_LIST]

    #Initialize lists
    for region_idx, _ in enumerate(REGION_KEYWORD_LIST):
        model_domain_avg_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list = []
        model_zonal_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list = []
        model_merid_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list = []
        model_pythag_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list = []
        model_fourier_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list = []
    
        smartinit_domain_avg_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list = []
        smartinit_zonal_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list = []
        smartinit_merid_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list = [] 
        smartinit_pythag_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list = []
        smartinit_fourier_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list = []

    #Calculate stats for any data that doesn't exist on disk
    for IDX in range(len(model_attrs.dataset)):
        predictor, model_output, target, dt_current = get_model_output_at_idx(model_attrs=model_attrs,
                                                                              model=model_attrs.model,
                                                                              idx=IDX,
                                                                              predictor_var=TARG_VAR,
                                                                              target_var=TARG_VAR)
    
        xr_smartinit = get_smartinit_output_at_idx(idx=IDX, target_var=TARG_VAR)
    
        _, model_output_cropped, smartinit_cropped, target_cropped = crop_to_intersection_of_inputs(predictor, model_output, xr_smartinit.data, target)
        
        for region_idx, region_keyword in enumerate(REGION_KEYWORD_LIST): #this may be the worst piece of code I've ever written but at least it works
            mo_r = restrict_to_region(model_output_cropped, region_keyword=region_keyword)
            t_r = restrict_to_region(target_cropped, region_keyword=region_keyword)
            sm_r = restrict_to_region(smartinit_cropped, region_keyword=region_keyword)
            
            model_grad_diff_zonal = calc_gradient_difference(mo_r, t_r, is_zonal=True)
            model_grad_diff_merid = calc_gradient_difference(mo_r, t_r, is_zonal=False)
            smartinit_grad_diff_zonal = calc_gradient_difference(sm_r, t_r, is_zonal=True)
            smartinit_grad_diff_merid = calc_gradient_difference(sm_r, t_r, is_zonal=False)
            
            region_abbreviation = C.region_keyword_dict[region_keyword]['abbreviation']
            
            data_savename_model_domain_avg_rmse_regional = f"{model_attrs.savename}_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
            data_savename_model_zonal_gradient_rmse_regional = f"{model_attrs.savename}_zonal_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
            data_savename_model_merid_gradient_rmse_regional = f"{model_attrs.savename}_merid_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
            data_savename_model_pythag_gradient_rmse_regional = f"{model_attrs.savename}_pythag_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
            data_savename_model_fourier_rmse_regional = f"{model_attrs.savename}_fourier_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
            
            data_savename_smartinit_domain_avg_rmse_regional = f"smartinit_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
            data_savename_smartinit_zonal_gradient_rmse_regional = f"smartinit_zonal_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
            data_savename_smartinit_merid_gradient_rmse_regional = f"smartinit_merid_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
            data_savename_smartinit_pythag_gradient_rmse_regional = f"smartinit_pythag_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
            data_savename_smartinit_fourier_rmse_regional = f"smartinit_fourier_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
            
            if not os.path.exists(f"{C.DIR_STATS_MODEL}/{data_savename_model_domain_avg_rmse_regional}"):
                model_domain_avg_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list.append(calc_RMSE(mo_r, t_r))
            if not os.path.exists(f"{C.DIR_STATS_MODEL}/{data_savename_model_zonal_gradient_rmse_regional}"):
                model_zonal_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list.append(calc_RMSE(model_grad_diff_zonal))
            if not os.path.exists(f"{C.DIR_STATS_MODEL}/{data_savename_model_merid_gradient_rmse_regional}"):
                model_merid_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list.append(calc_RMSE(model_grad_diff_merid))
            if not os.path.exists(f"{C.DIR_STATS_MODEL}/{data_savename_model_pythag_gradient_rmse_regional}"):
                model_pythag_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list.append(calc_RMSE(np.hypot(model_grad_diff_zonal, model_grad_diff_merid)))
            if not os.path.exists(f"{C.DIR_STATS_MODEL}/{data_savename_model_fourier_rmse_regional}"):
                model_fourier_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list.append(calc_fourier_RMSE(mo_r, t_r))

            if not os.path.exists(f"{C.DIR_STATS_SMARTINIT}/{data_savename_smartinit_domain_avg_rmse_regional}"):
                smartinit_domain_avg_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list.append(calc_RMSE(sm_r, t_r))
            if not os.path.exists(f"{C.DIR_STATS_SMARTINIT}/{data_savename_smartinit_zonal_gradient_rmse_regional}"):
                smartinit_zonal_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list.append(calc_RMSE(smartinit_grad_diff_zonal))
            if not os.path.exists(f"{C.DIR_STATS_SMARTINIT}/{data_savename_smartinit_merid_gradient_rmse_regional}"):
                smartinit_merid_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list.append(calc_RMSE(smartinit_grad_diff_merid))
            if not os.path.exists(f"{C.DIR_STATS_SMARTINIT}/{data_savename_smartinit_pythag_gradient_rmse_regional}"):
                smartinit_pythag_gradient_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list.append(calc_RMSE(np.hypot(smartinit_grad_diff_zonal, smartinit_grad_diff_merid)))
            if not os.path.exists(f"{C.DIR_STATS_SMARTINIT}/{data_savename_smartinit_fourier_rmse_regional}"):
                smartinit_fourier_rmse_regional_statsobjs_list[region_idx].domain_avg_rmse_alltimes_list.append(calc_fourier_RMSE(sm_r, t_r))

        if IDX%int(len(model_attrs.dataset)/100)==0:
            print(f"{(IDX/len(model_attrs.dataset))*100:.0f}% done")

    #Save all newly generated data
    for region_idx, region_keyword in enumerate(REGION_KEYWORD_LIST): 
        region_abbreviation = C.region_keyword_dict[region_keyword]['abbreviation']
        
        data_savename_model_domain_avg_rmse_regional = f"{model_attrs.savename}_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
        data_savename_model_zonal_gradient_rmse_regional = f"{model_attrs.savename}_zonal_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
        data_savename_model_merid_gradient_rmse_regional = f"{model_attrs.savename}_merid_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
        data_savename_model_pythag_gradient_rmse_regional = f"{model_attrs.savename}_pythag_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
        data_savename_model_fourier_rmse_regional = f"{model_attrs.savename}_fourier_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
        
        data_savename_smartinit_domain_avg_rmse_regional = f"smartinit_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
        data_savename_smartinit_zonal_gradient_rmse_regional = f"smartinit_zonal_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
        data_savename_smartinit_merid_gradient_rmse_regional = f"smartinit_merid_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
        data_savename_smartinit_pythag_gradient_rmse_regional = f"smartinit_pythag_gradient_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
        data_savename_smartinit_fourier_rmse_regional = f"smartinit_fourier_RMSE_alltimes_{region_abbreviation}_{TARG_VAR}.csv"
        
        save_rmse_data_to_disk(model_domain_avg_rmse_regional_statsobjs_list.domain_avg_rmse_alltimes_list[region_idx], data_savename_model_domain_avg_rmse_regional)
        save_rmse_data_to_disk(model_zonal_gradient_rmse_regional_statsobjs_list.domain_avg_rmse_alltimes_list[region_idx], data_savename_model_zonal_gradient_rmse_regional)
        save_rmse_data_to_disk(model_merid_gradient_rmse_regional_statsobjs_list.domain_avg_rmse_alltimes_list[region_idx], data_savename_model_merid_gradient_rmse_regional)
        save_rmse_data_to_disk(model_pythag_gradient_rmse_regional_statsobjs_list.domain_avg_rmse_alltimes_list[region_idx], data_savename_model_pythag_gradient_rmse_regional)
        save_rmse_data_to_disk(model_fourier_rmse_regional_statsobjs_list.domain_avg_rmse_alltimes_list[region_idx], data_savename_model_fourier_rmse_regional)

        save_rmse_data_to_disk(smartinit_domain_avg_rmse_regional_statsobjs_list.domain_avg_rmse_alltimes_list[region_idx], data_savename_smartinit_domain_avg_rmse_regional)
        save_rmse_data_to_disk(smartinit_zonal_gradient_rmse_regional_statsobjs_list.domain_avg_rmse_alltimes_list[region_idx], data_savename_smartinit_zonal_gradient_rmse_regional)
        save_rmse_data_to_disk(smartinit_merid_gradient_rmse_regional_statsobjs_list.domain_avg_rmse_alltimes_list[region_idx], data_savename_smartinit_merid_gradient_rmse_regional)
        save_rmse_data_to_disk(smartinit_pythag_gradient_rmse_regional_statsobjs_list.domain_avg_rmse_alltimes_list[region_idx], data_savename_smartinit_pythag_gradient_rmse_regional)
        save_rmse_data_to_disk(smartinit_fourier_rmse_regional_statsobjs_list.domain_avg_rmse_alltimes_list[region_idx], data_savename_smartinit_fourier_rmse_regional)

    print(f"!!!!!!! ALL DATA FOR {model_attrs.savename} WRITTEN TO DISK !!!!!!!")
        