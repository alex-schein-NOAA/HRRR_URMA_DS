from HEADER_torch import *




########################################################
### DATA FETCHING FUNCTIONS
# Any functions that get data off disk or from models
########################################################

def get_model_output_at_idx(model_attrs, 
                            model, 
                            pred_var="t2m", 
                            targ_var="t2m", 
                            idx=0, 
                            is_nan=True,
                            nan_fill_value=0,
                            is_unnormed=True,
                            crop_pred=False,
                            crop_model_output=False,
                            crop_targ=False,
                            device="cuda"
                           ):
    """
    Inputs:
        - model_attrs --> DefineModelAttributes object. MUST HAVE .create_dataset() ALREADY CALLED! 
        - model --> Pytorch model to use, with weights loaded and device initialized
        - pred_var --> string of the predictor variable to get the output of. See the dataset class for valid options
        - targ_var --> string of the target variable to get the output of
        - idx --> int, index to get the output of (time index)
        - is_nan --> bool to control if the model should be applied to NaN data (if False) or if the NaN data should be replaced (if True [default]). 
            > Should generally be set to True if predictor data has NaNs (e.g. CONUS HRRR data) because the models don't apply to NaNs and thus the output is severely truncated from what it should be. 
            
        - nan_fill_value: int or float, used to fill in all NaN values in the predictor data, if is_nan=True
        - is_unnormed: bool; if True (default), returns unnormed data. Predictor and target data is read directly from their raw xarray files, whilst model_output is unnormalized by the corresponding variable's stored (pseudo-) mean and stddev
        - crop_pred: bool, default value = False; if True, crops the predictor data to cut out regions of NaNs for better plotting. This also serves as the mask for crop_model_output and crop_targ
        - crop_model_output: bool, default value = False; if True (default), crops the model output data according to the predictor var's mask
        - crop_targ: bool, default value = False; if True (default), crops the target data " "
        - device: cuda device, default to just "cuda". Might need to change this in calling function, be careful

    Outputs:
        - predictor @ index, UNNORMED if is_unnormed, CROPPED if crop_pred
        - target @ index, UNNORMED if is_unnormed, CROPPED if crop_model_output
        - model output @ index, UNNORMED if is_unnormed, CROPPED if crop_targ
        - dt_current as dt.datetime object, for plot title purposes
    """
    
    pred,targ = model_attrs.dataset[idx]
    if is_nan: #Added 2025-10-15
        np.nan_to_num(pred, copy=False, nan=nan_fill_value) 
    pred = pred[np.newaxis,:] 
    pred_gpu = torch.from_numpy(pred).cuda(device)
    
    with torch.no_grad():
        model_output = model(pred_gpu.float())
        model_output = model_output.cpu().numpy()
    
    date = model_attrs.dataset.xr_datasets_pred[model_attrs.predictor_vars.index(pred_var)][idx].valid_time.data
    dt_current = dt.datetime.strptime(str(np.datetime_as_string(date, unit='m')), "%Y-%m-%dT%H:%M")
    
    if is_unnormed:
        pred = model_attrs.dataset.xr_datasets_pred[model_attrs.predictor_vars.index(pred_var)][idx].data
        targ = model_attrs.dataset.xr_datasets_targ[model_attrs.target_vars.index(targ_var)][idx].data

        model_output = ( model_attrs.dataset.datasets_targ_normed_stddevs[model_attrs.target_vars.index(targ_var)]
                         *model_output[0,model_attrs.target_vars.index(targ_var),:] 
                         + model_attrs.dataset.datasets_targ_normed_means[model_attrs.target_vars.index(targ_var)] )
    
    else: #model output is already normed
        pred = pred[0,model_attrs.predictor_vars.index(pred_var),:]
        targ = targ[model_attrs.target_vars.index(targ_var),:]

    if crop_model_output:
        model_output = crop_input(model_output, pred)
    if crop_targ:
        targ = crop_input(targ, pred)
    if crop_pred: #pred done last because it first has to serve as the mask for the previous data
        pred = crop_input(pred, pred)
    
    return pred, targ, model_output, dt_current