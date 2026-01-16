from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *

from FunctionsAndClasses.CONSTANTS import *

######################################################################################################################################################

########################################################
### DATA FETCHING FUNCTIONS
# Any functions that get data off disk or from models
########################################################

def get_model_output_at_idx(model_attrs, 
                            model, 
                            predictor_var="t2m", 
                            target_var="t2m", 
                            idx=0, 
                            is_nan=True,
                            nan_fill_value=0,
                            is_unnormed=True,
                            device="cuda"
                           ):
    """
    Inputs:
        - model_attrs --> DefineModelAttributes object. MUST HAVE .create_dataset() ALREADY CALLED! 
        - model --> Pytorch model to use, with weights loaded and device initialized
        - predictor_var --> string of the predictor variable to get the output of. See the dataset class for valid options
        - target_var --> string of the target variable to get the output of
        - idx --> int, index to get the output of (time index)
        - is_nan --> bool to control if the model should be applied to NaN data (if False) or if the NaN data should be replaced (if True [default]). 
            > Should generally be set to True if predictor data has NaNs (e.g. CONUS HRRR data) because the models don't apply to NaNs and thus the output is severely truncated from what it should be. 
        - nan_fill_value --> int or float, used to fill in all NaN values in the predictor data, if is_nan=True. Default = 0 (best value, experimentally determined)
        - is_unnormed --> bool; if True (default), returns unnormed data. Predictor and target data is read directly from their raw xarray files, whilst model_output is unnormalized by the corresponding variable's stored (pseudo-) mean and stddev
        - device --> cuda device, default to just "cuda". Might need to change this in calling function if one GPU is overloaded

    Outputs:
        - predictor @ index, UNNORMED if is_unnormed, CROPPED if crop_predictor
        - model output @ index, UNNORMED if is_unnormed, CROPPED if crop_target
        - target @ index, UNNORMED if is_unnormed, CROPPED if crop_model_output
        - dt_current as dt.datetime object, for plot title purposes
    """
    
    predictor, target = model_attrs.dataset[idx]
    if is_nan: 
        np.nan_to_num(predictor, copy=False, nan=nan_fill_value) 
    predictor = predictor[np.newaxis,:] 
    predictor_gpu = torch.from_numpy(predictor).cuda(device)
    
    # with torch.no_grad():
    #     model_output = model(predictor_gpu.float())
    #     model_output = model_output.cpu().numpy()

    #Experimental, to free up GPU memory
    with torch.no_grad():
        model_output_gpu = model(predictor_gpu.float())
        model_output = model_output_gpu.cpu().numpy()
    del predictor_gpu, model_output_gpu
    torch.cuda.empty_cache()
    
    date = model_attrs.dataset.xr_datasets_predictor[model_attrs.predictor_vars.index(predictor_var)][idx].valid_time.data
    dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='m')), "%Y-%m-%dT%H:%M")
    
    if is_unnormed:
        predictor = model_attrs.dataset.xr_datasets_predictor[model_attrs.predictor_vars.index(predictor_var)][idx].data
        target = model_attrs.dataset.xr_datasets_target[model_attrs.target_vars.index(target_var)][idx].data

        model_output = ( model_attrs.dataset.datasets_target_normed_stddevs[model_attrs.target_vars.index(target_var)]
                         *model_output[0,model_attrs.target_vars.index(target_var),:] 
                         + model_attrs.dataset.datasets_target_normed_means[model_attrs.target_vars.index(target_var)] )
    
    else: #model output is already normed
        predictor = predictor[0,model_attrs.predictor_vars.index(predictor_var),:]
        target = target[model_attrs.target_vars.index(target_var),:]
    
    return predictor, model_output, target, dt_current

########################################################

def get_smartinit_output_at_idx(idx, 
                                target_var,
                                FORECAST_LEAD_HOURS=1, 
                                smartinit_directory=None,
                                smartinit_var_select_dict=None, 
                                varname_translation_dict=None, 
                                START_DATE=None
                               ):
    """
    Method to open one Smartinit file and return its output for one variable, restricted to whatever spatial domain we define.
    Designed for StatObjectConstructor but can be called from anywhere else that Smartinit output is needed.

    Inputs:
        - idx --> int of index to select. Should line up with sample_idx indexing from HRRR
        - target_var --> string of a valid target variable, e.g. "t2m"
        - FORECAST_LEAD_HOURS --> int of forecast lead time. Default = 1. 
            > !!! Should already have offset START_DATE if START_DATE is not None !!!
            > Should never be changed from default - only included to extend functionality if needed
        - smartinit_directory --> string of directory of smartinit data which is NOT subset in any way but is named according to the convention in the code below. 
            > If None (default), autodirects to the smartinit directory in the CONSTANTS class.
        - smartinit_var_select_dict --> as in CONSTANTS. 
            > If None (default), autodirects to the appropriate dict in the CONSTANTS class.
        - varname_translation_dict --> as in CONSTANTS. 
            > If None (default), autodirects to the appropriate dict in the CONSTANTS class.
        - START_DATE --> dt.datetime object. Currently we only have Smartinit data for 2024, so this should be 2023/12/31 23z or later. 
            > Should pretty much never be changed from the default value
            > !!! VERY IMPORTANT: if not None, then the calling function should have START_DATE = dt.datetime([20240101_00z or greater])-dt.timedelta(hours=FORECAST_LEAD_HOURS) !!!

    Outputs:
        - xr_smartinit --> xarray object of the smartinit data @ i + target_var
            - Returns xarray object, not just data, so calling function should invoke .data if that's what's desired
    """

    C = CONSTANTS()

    #These should generally NOT be changed in the function call
    if smartinit_directory is None:
        smartinit_directory = C.DIR_SMARTINIT_DATA
    if smartinit_var_select_dict is None:
        smartinit_var_select_dict = C.urma_var_select_dict #they share the same keys
    if varname_translation_dict is None:
        varname_translation_dict = C.varname_translation_dict
    
    if START_DATE is None: #This should be the default, as this method is written to count hours from the first available Smartinit date, which is 2024-01-01 00z
        START_DATE = dt.datetime(2024,1,1,0)-dt.timedelta(hours=FORECAST_LEAD_HOURS)
    
    DATE_STR = dt.date.strftime(START_DATE + dt.timedelta(hours=idx), "%Y%m%d")
    file_to_open = f"{smartinit_directory}/hrrr_smartinit_{DATE_STR}_t{str((START_DATE.hour+idx)%24).zfill(2)}z_f{str(FORECAST_LEAD_HOURS).zfill(2)}_regridded.grib2" #changed 2025-10-22 to read the regridded CONUS smartinit
    xr_smartinit = xr.open_dataset(file_to_open,
                                   engine="cfgrib", 
                                   backend_kwargs=smartinit_var_select_dict[varname_translation_dict[target_var]],
                                   decode_timedelta=True)
    xr_smartinit = xr_smartinit[varname_translation_dict[target_var]]
    
    return xr_smartinit

########################################################

def get_valid_region_mask(VALID_REGION_FILEPATH=None, 
                          PATCH_SIZE=None, 
                          file_save_dir=None, 
                          file_save_name=None):
    """
    Function to either fetch a precalculated mask of valid locations for patch's SW corner selection for use in the CONUS Dataset class, or calculate and return+save that data if it doesn't exist in file_save_dir.
    SAVED FILES SHOULD/WILL BE OF DIMENSION 1597x2345, i.e. the entire extended HRRR/URMA domain, because this is what the Dataset class(es) use and what __getitem__ will expect!

    Inputs:
        - VALID_REGION_FILEPATH = full filepath (including extension - should be csv) to the file containing the binary valid/invalid location data, which should be formatted according to this function's writing functionality. If it exists, it's read in and returned, otherwise it's calculated then returned + saved to disk
            > If None, defaults to {file_save_dir}/{file_save_name}
        - PATCH_SIZE - int for square patch size. If None (default), then the PATCH_SIZE in the CONSTANTS class is used. Should usually be included in the calling function's use, just in case
        - file_save_dir = full filepath to the save directory, if not using the default location
            > Default location = /scratch3/BMC/wrfruc/aschein/Train_Test_Files/
        - file_save_name = savename (including .csv extension) of the file, if not using the default construction
            > Default savename = "valid_region_for_patch_sw_corner_PATCHSIZE{PATCH_SIZE}.csv" - this only works for square patches, so revisit this convention if nonsquare patches are used!
    """

    C = CONSTANTS()
    
    if PATCH_SIZE is None:
        PATCH_SIZE = C.PATCH_SIZE 

    if file_save_name is None:
        file_save_name = f"valid_region_for_patch_sw_corner_PATCHSIZE{PATCH_SIZE}.csv"

    if file_save_dir is None:
        file_save_dir = C.DIR_TRAIN_TEST

    if VALID_REGION_FILEPATH is None:
        VALID_REGION_FILEPATH = f"{file_save_dir}/{file_save_name}"
    
    if os.path.exists(VALID_REGION_FILEPATH): #read in the data 
        valid_region_for_patch_sw_corner = np.genfromtxt(VALID_REGION_FILEPATH, delimiter=',')
    else: #need to calculate the region, and save to disk then return
        print(f"Valid patch selection region for PATCH_SIZE = {PATCH_SIZE} doesn't exist on disk (should be at {VALID_REGION_FILEPATH}). Computing this region now")
        xr_hrrr_conus = xr.open_dataarray(f"{C.DIR_TRAIN_TEST}/test_hrrr_alltimes_CONUS_t2m_f01.grib2", decode_timedelta=True, engine='cfgrib') #using t2m as the default - doesn't matter
        data_mask = xr_hrrr_conus[0].notnull() #Somewhat bugged if calling this in the below selection, so it needs to be separate
        xr_hrrr_conus_masked = xr_hrrr_conus[0].where(data_mask, drop=False) #don't drop! Need to keep the dimension
        valid_region_for_patch_sw_corner =  np.zeros((np.shape(xr_hrrr_conus_masked.data)))
        
        start = time.time()
        for lat_idx in xr_hrrr_conus_masked.y.data[:-PATCH_SIZE]: #Can't extend a patch beyond the domain!
            for lon_idx in xr_hrrr_conus_masked.x.data[:-PATCH_SIZE]: #Can't extend a patch beyond the domain!
                patch = xr_hrrr_conus_masked[lat_idx:lat_idx+PATCH_SIZE, lon_idx:lon_idx+PATCH_SIZE]
                if ~np.any(patch.isnull()): #if the patch is ok, flip the corresponding index in valid_region_for_patch_sw_corner
                    valid_region_for_patch_sw_corner[lat_idx, lon_idx]=1

        with open(VALID_REGION_FILEPATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(valid_region_for_patch_sw_corner) 
        
        print(f"Valid region computed, time taken = {time.time() - start:.1f} seconds. New file located at {VALID_REGION_FILEPATH} for future use")

    return valid_region_for_patch_sw_corner

########################################################

def check_rmse_data_and_read_in(data_savename, suppress_printout=False):
    """
    Checks if precalculated RMSE data (as specified by data_savename) exists on disk. 
    If True, reads it into a new list and returns that list. 
    If False, gives a warning and returns an empty list. 

    Calling function is responsible for data sanitization!
    """
    C = CONSTANTS()

    if ".csv" not in data_savename:
        data_savename = data_savename + ".csv"
    
    if "pred(" in data_savename: #horrid hack but works for current savename scheme
        if os.path.exists(f"{C.DIR_STATS_MODEL}/{data_savename}"):
            if not suppress_printout:
                print(f"{data_savename} exists on disk. Reading in now...")
            with open(f"{C.DIR_STATS_MODEL}/{data_savename}", 'r', newline='') as file:
                reader = csv.reader(file)
                rmse_list = [np.float32(x) for x in (list(reader))[0]]
            return rmse_list
        else:
            print(f"{data_savename} does not exist on disk.")
            return []
            
    elif "smartinit" in data_savename:
        if os.path.exists(f"{C.DIR_STATS_SMARTINIT}/{data_savename}"):
            if not suppress_printout:
                print(f"{data_savename} exists on disk. Reading in now...")
            with open(f"{C.DIR_STATS_SMARTINIT}/{data_savename}", 'r', newline='') as file:
                reader = csv.reader(file)
                rmse_list = [np.float32(x) for x in (list(reader))[0]]
            return rmse_list
        else:
            print(f"{data_savename} does not exist on disk.")
            return []
    
    else:
        print(f"data_savename doesn't contain the proper keyword(s) to distinguish model stats from smartinit stats. Retry!")
        return []

########################################################

def save_rmse_data_to_disk(rmse_list, data_savename):
    """
    Takes in list of RMSE data and a corresponding savename and saves it to the appropriate directory.
    """
    C = CONSTANTS()

    if "pred(" in data_savename:
        if not os.path.exists(f"{C.DIR_STATS_MODEL}/{data_savename}"):
            with open(f"{C.DIR_STATS_MODEL}/{data_savename}", "w", newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(rmse_list)
            print(f"{data_savename} written to {C.DIR_STATS_MODEL}")
        else:
            print(f"{data_savename} already exists in {C.DIR_STATS_MODEL}")

    elif "smartinit" in data_savename:
        if not os.path.exists(f"{C.DIR_STATS_SMARTINIT}/{data_savename}"):
            with open(f"{C.DIR_STATS_SMARTINIT}/{data_savename}", "w", newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(rmse_list)
            print(f"{data_savename} written to {C.DIR_STATS_SMARTINIT}")
        else:
            print(f"{data_savename} already exists in {C.DIR_STATS_SMARTINIT}")

    else:
        print(f"data_savename doesn't contain the proper keyword(s) to distinguish model stats from smartinit stats. Retry!")
        
    return
    
######################################################################################################################################################

########################################################
### DATA CROPPING/MANIPULATION FUNCTIONS
# Any functions that manipulate or crop data 
########################################################

def crop_input(data_to_crop, data_for_mask, return_xr=False):
    """
    Crops out areas of only NaNs. Note this takes 2 input arguments, which can be the same but usually aren't for cropping over CONUS...
    Inputs:
    - data_to_crop --> input 2D numpy array that we want cropped. Does NOT need to have regions of NaNs! Can also be an xarray object (usually if return_xr=True)
    - data_for_mask --> input 2D numpy array that DOES have regions of NaNs, whose mask is to be used to crop data_to_crop. Can also be an xarray object (usually if return_xr=True)
    - return_xr --> bool to determine if the resulting xarray objects are returned or not. Default = False, but can be set to True if feeding in xarray objects whose properties should be preserved

    Output:
        - Cropped version of data_to_crop; raw array if return_xr=False, or xarray object if True
    
    Usual use case: data_for_mask = predictor data for HRRR, data_to_crop = whatever other data (could also be predictor data, or model output, or target data)
    
    """
    data_to_crop_xr = xr.DataArray(data_to_crop, dims=('y','x')) #Easiest to cast in xarray. (y,x) should be the ordering of the dims for such data
    data_for_mask_xr = xr.DataArray(data_for_mask, dims=('y','x')) #Also needs to be a DataArray for .where to work
    data_to_crop_xr_cropped = data_to_crop_xr.where(~np.isnan(data_for_mask_xr), drop=True)

    if return_xr:
        return data_to_crop_xr_cropped
    else:
        return data_to_crop_xr_cropped.data #return raw array, not xarray object

########################################################

def crop_to_intersection_of_inputs(predictor,
                                   model_1_output,
                                   model_2_output,
                                   target=None,
                                   return_xr=False
                                   ):
    """ 
    Function to do restriction of data to a shared domain between model_1_output (assumed to have the same region as predictor) and model_2_output.
    Intended use: model_1_output = ML model output (on the HRRR region), model_2_output = Smartinit

    !! Should be used INSTEAD OF crop_input - i.e. on arrays that have NOT yet been spatially restricted !!

    Inputs: all of the following should be arrays, but could be xr objects (uncommon)
        - predictor --> array of predictor (i.e. HRRR) data, with surrounding region of NaNs. MUST BE INCLUDED
        - model_1_output --> array of model output (should be ML model, not Smartinit). Does not need to have surrounding NaNs. MUST BE INCLUDED
        - model_2_output --> array of model output; intended to be Smartinit, but could be a second ML model's output, in which case this function will just return the same thing as crop_input, but for all arrays involved. MUST BE INCLUDED
        - target --> array of target (i.e. URMA). OPTIONAL - though should usually be included
        - return_xr --> bool to control if raw arrays are returned (if False, which is default) or xarray objects (if True). In the latter case, all the inputs must also be xarray objects!
        
    Outputs: 
        - Restricted to their common intersection, in order: 
            predictor, model_1_output, model_2_output [, target if not None]
    """

    # First restrict to the HRRR domain
    if target is not None:
        target = crop_input(target, predictor, return_xr=return_xr)
    model_1_output = crop_input(model_1_output, predictor, return_xr=return_xr)
    model_2_output = crop_input(model_2_output, predictor, return_xr=return_xr)
    predictor = crop_input(predictor, predictor, return_xr=return_xr)

    # Then restrict to model_2_output's domain - will restrict futher if this is Smartinit, but will not do anything if model_2_output is also on the HRRR domain
    if target is not None:
        target = crop_input(target, model_2_output, return_xr=return_xr)
    predictor = crop_input(predictor, model_2_output, return_xr=return_xr)
    model_1_output = crop_input(model_1_output, model_2_output, return_xr=return_xr)
    model_2_output = crop_input(model_2_output, model_2_output, return_xr=return_xr)

    if target is not None:
        return predictor, model_1_output, model_2_output, target
    else:
        return predictor, model_1_output, model_2_output

########################################################

def restrict_to_region(data, 
                       RESTR_ORIGIN_LAT_IDX=None,
                       RESTR_ORIGIN_LON_IDX=None,
                       RESTR_PATCH_SIZE_LAT=None,
                       RESTR_PATCH_SIZE_LON=None,
                       region_keyword=None
                      ):
    """ 
    Restricts input data (i.e. an array or xr object) to the defined region. Data must be HRRR/URMA/model output/Smartinit formatted, i.e. origin @ SW corner of the domain.
    Stores some commonly used domains for data WHICH HAS ALREADY BEEN RESTRICTED to the intersection of the HRRR and Smartinit regions, i.e. data that is 1358 x 2145. 
    !! Calling function must do this restriction beforehand !!! Use crop_to_intersection_of_inputs

    Inputs:
        - data --> input data to be restricted. Must be formatted as above
        - RESTR_ORIGIN_[LAT/LON]_IDX --> ints to define the SW corner lat/lon 
        - RESTR_PATCH_SIZE_[LAT/LON] --> ints to define the north/east extent of the selection, respectively
        - region_keyword --> string (case-sensitive) of the region to select, if any. Valid options as follows:
            > "Colorado" --> 200x200 patch over the Colorado Rockies
            > "California" --> 350x200 lat/lon patch over California (and a bit of the surrounding area)
            > "West" --> 750x750 patch over the western US (designed to be close to the MPAS west domain)
            > "Appalachia" --> 400x330 lat/lon patch over the Appalachian mountains and surrounding areas 
            > "Great Lakes" --> 400x600 lat/lon patch over the Great Lakes region
            > "Northeast" --> 450x450 patch over the Northeast, roughly bottom of PA to top of ME

    Output:
        - data, restricted to whatever region is defined
    """
    C = CONSTANTS()

    if not region_keyword==None:
        RESTR_ORIGIN_LAT_IDX = C.region_keyword_dict[region_keyword]['RESTR_ORIGIN_LAT_IDX']
        RESTR_ORIGIN_LON_IDX = C.region_keyword_dict[region_keyword]['RESTR_ORIGIN_LON_IDX']
        RESTR_PATCH_SIZE_LAT = C.region_keyword_dict[region_keyword]['RESTR_PATCH_SIZE_LAT']
        RESTR_PATCH_SIZE_LON = C.region_keyword_dict[region_keyword]['RESTR_PATCH_SIZE_LON']
    
    return data[RESTR_ORIGIN_LAT_IDX:RESTR_ORIGIN_LAT_IDX+RESTR_PATCH_SIZE_LAT, RESTR_ORIGIN_LON_IDX:RESTR_ORIGIN_LON_IDX+RESTR_PATCH_SIZE_LON]