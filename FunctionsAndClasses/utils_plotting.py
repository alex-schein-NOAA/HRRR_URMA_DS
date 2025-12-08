from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HEADER_plotting import *

from FunctionsAndClasses.CONSTANTS import *
from FunctionsAndClasses.utils_data import *
from FunctionsAndClasses.utils_miscellaneous import *

######################################################################################################################################################

########################################################
### PLOTTING FUNCTIONS
# Any functions that plot data. Does not include helper functions
########################################################
    
def plot_predictor_output_truth_error(predictor, 
                                      model_output, 
                                      target, 
                                      date_str="DATE", 
                                      title="MODEL_NAME", 
                                      save_fig=False, 
                                      save_dir=os.getcwd(), 
                                      fig_savename="temp.png", 
                                      error_units="", 
                                      avg_denom=10
                                     ):
    """
    Plots the 1x4 figure for spatial display of [input | prediction | truth | (output-truth)], given a single hour's worth of data.
    Input data should generally be obtained from get_model_output_at_idx and/or get_smartinit_output_at_idx
    
    Inputs:
        - predictor --> array of predictor input (i.e. 2.5 km HRRR for our purposes)
        - model_output --> array of model output, or whatever else (e.g. Smartinit field)
        - target --> array of "truth" data (i.e. URMA for our purposes)
        - date_str --> string or datetime object of format dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S") 
        - title --> model name/params/whatever to identify that plot
        - save_fig --> bool; if True, saves to save_dir 
        - save_dir --> string or filepath to master save directory. Default = current working directory
        - fig_savename --> string for file savename, if save_fig = True. Should include ".png"
        - error_units --> string for error units, e.g. "deg K" (usually is f"{C.varname_units_dict[TARG_VAR]}")
        - avg_denom --> how much to scale error plot by. Should be ~10 for temperature/wind vars, but for pressurf, should be ~150
    """

    #predictor, model_output, target = input data, model prediction, truth, respectively, as numpy arrays
    fig, axes = plt.subplots(1,4, figsize=(20,5))
    maxtemp = np.nanmax([np.nanmax(predictor.squeeze()), np.nanmax(model_output.squeeze()), np.nanmax(target.squeeze())])
    mintemp = np.nanmin([np.nanmin(predictor.squeeze()), np.nanmin(model_output.squeeze()), np.nanmin(target.squeeze())])

    avg = (maxtemp-mintemp)/avg_denom #Denominator chosen arbitrarily; adjust if needed
    
    axes[0].imshow(predictor.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
    axes[0].set_title(f"Predictor (HRRR 2.5km)")
    axes[0].axis("off")
    axes[1].imshow(model_output.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
    axes[1].set_title(f"Predicted")
    axes[1].axis("off")
    axes[2].imshow(target.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
    axes[2].set_title(f"Truth (URMA)")
    axes[2].axis("off")
    pos = axes[3].imshow((model_output.squeeze() - target.squeeze()), cmap="bwr", vmin = -1*avg, vmax = avg, origin='lower') 
    axes[3].set_title(f"Prediction - Truth (RMSE = {np.sqrt(np.nanmean((model_output.squeeze() - target.squeeze())**2)):.3f})")
    axes[3].axis("off")

    cbar = fig.colorbar(pos, ax=axes[3], fraction=0.03) 
    cbar.set_label(f'Error ({error_units})')
    
    plt.suptitle(f"{title} | Date = {date_str} \n Maximum = {maxtemp:.1f} | Minimum = {mintemp:.1f}", va="bottom", fontsize=14)
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"{save_dir}/{fig_savename}",dpi=300, bbox_inches="tight")

    plt.show()
    return

########################################################

def plot_predictor_output_truth_error_CONUS(predictor, 
                                            model_output,
                                            target,
                                            include_predictor=False,
                                            include_model_output=False,
                                            include_target=False,
                                            use_hrrr_mask=True,
                                            use_smartinit_mask=False,
                                            date_str="DATE",
                                            title="MODEL NAME",
                                            error_units="",
                                            save_fig=False, 
                                            save_dir=f"/scratch3/BMC/wrfruc/aschein/UNet_main", 
                                            fig_savename="temp.png", 
                                            avg_denom=10
                                           ):

    """
    NOTE: model error (vs target) will ALWAYS be included - predictor, model output, and target are all optional plots! However, they are NOT optional data to include!
    Output plot's shape will depend on the number of fields included, up to 4 plots
    Order of plots (if included) = predictor, model output, truth, error (as before)

    This is not a very well constructed function, and there has to be a more refined and better way to do the dynamic plots, but this works for now
    
    Inputs:
        - predictor --> predictor input (i.e. 2.5 km HRRR for our purposes) from get_model_output
            > If using this function to plot Smartinit, make sure the call to get_model_output_at_idx has crop_pred=False 
        - model_output --> Model output data from get_model_output. CAN ALSO BE SMARTINIT DATA in which case, make sure use_smartinit_mask=True
            > If using this function to plot Smartinit, make sure the call to get_model_output_at_idx has crop_model_output=False 
        - target --> truth (i.e. URMA for our purposes) from get_model_output
            > If using this function to plot Smartinit, make sure the call to get_model_output_at_idx has crop_targ=False 
        - include_predictor --> bool to control if a plot showing the predictor data is included
        - include_model_output --> " " for model output data
        - include_target --> " " for target data
        - use_hrrr_mask --> bool to control if a mask to exclude the NaN regions of the regridded HRRR data will be used. 
            > For Smartinit, it should pretty much always be True, if the goal is to compare Smartinit against HRRRR
        - use_smartinit_mask --> " " for regridded Smartinit data. Doesn't always need to be True, but should be if comparing HRRR against Smartinit
        - date_str --> string of format dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S") from get_model_output()
        - title --> model name/params/whatever to identify that plot
        - save_fig --> bool for saving; if True, saves to directory this script is called from (currently this function is not intended for formalized plot saving)
        - save_dir --> master save directory
        - fig_savename --> string for file savename, if to_save = True. Should include ".png" at the end
        - error_units --> string (NOT including parentheses) for variable/error units, e.g. "deg K" (usually is f"{C.varname_units_dict[TARG_VAR]}")
        - avg_denom --> int for how much to scale error plot by. Should be ~10 normally, but for pressurf, should be ~150
    """
    
    
    if use_smartinit_mask and use_hrrr_mask: 
        #Plotting only the overlap region. Here, predictor serves as the HRRR mask, but model_output may not be Smartinit data, so a new instance of Smartinit is called to be safe
        xr_smartinit = get_smartinit_output_at_idx(i=0, target_var='t2m') #only need the mask, don't care about the data
        predictor, model_output, _, target = crop_to_intersection_of_inputs(predictor, model_output, xr_smartinit.data, target)
    else:
        # This works for the cases of model_output being model output or Smartinit, and we only want to crop to those respective regions
        # Note that this technically excludes the case of use_hrr_mask=False && use_smartinit_mask=False, but in practice we never want to do this anyway, as this would plot the larger URMA domain with a lot of NaNs. If this ever needs to be implemented, make a new case for it
        predictor, model_output, _, target = crop_to_intersection_of_inputs(predictor, model_output, model_output, target)

    number_of_plots = 1+int(include_predictor)+int(include_model_output)+int(include_target) 
    maxtemp = np.nanmax([np.nanmax(predictor.squeeze()), np.nanmax(model_output.squeeze()), np.nanmax(target.squeeze())])
    mintemp = np.nanmin([np.nanmin(predictor.squeeze()), np.nanmin(model_output.squeeze()), np.nanmin(target.squeeze())])

    avg = (maxtemp-mintemp)/avg_denom #Denominator chosen arbitrarily; adjust if needed
    
    #Break it down into cases. This flag structure is horrible and should be redone
    pred_flag = True
    model_flag = True
    
    if number_of_plots > 1:
        fig, axes = plt.subplots(number_of_plots, 1, figsize=(10, 5.5*number_of_plots))
        for i, ax in enumerate(axes[:-1]):
            if include_predictor and pred_flag:
                pos = ax.imshow(predictor.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
                ax.set_title(f"Predictor (HRRR 2.5km)")
                cbar = fig.colorbar(pos, ax=ax, fraction=0.0225, pad=0.01)
                cbar.set_label(f'{error_units}')
                pred_flag = False #skip this case in the next iteration, if there is one
            elif include_model_output and model_flag:
                pos = ax.imshow(model_output.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
                ax.set_title(f"Predicted")
                cbar = fig.colorbar(pos, ax=ax, fraction=0.0225, pad=0.01)
                cbar.set_label(f'{error_units}')
                model_flag = False
            elif include_target: #No flag needed here, as this will always be the last plot, i.e. end of for loop
                pos = ax.imshow(target.squeeze(), cmap="coolwarm", vmin = mintemp, vmax = maxtemp, origin='lower')
                ax.set_title(f"Truth (URMA)")
                cbar = fig.colorbar(pos, ax=ax, fraction=0.0225, pad=0.01)
                cbar.set_label(f'{error_units}')

            ax.axis("off")
        plt.suptitle(f"{title} \n Date = {date_str} \n Maximum = {maxtemp:.1f} | Minimum = {mintemp:.1f}", va="bottom", fontsize=14)
                
        #Always make the last plot the error
        pos = axes[-1].imshow((model_output.squeeze() - target.squeeze()), cmap='bwr', origin='lower', vmin=-1*avg, vmax=avg)
        axes[-1].set_title(f"Prediction - Truth (RMSE = {np.sqrt(np.nanmean((model_output.squeeze() - target.squeeze())**2)):.3f})")
        axes[-1].axis("off")
        cbar = fig.colorbar(pos, ax=axes[-1], fraction=0.0225, pad=0.01)
        cbar.set_label(f'Error ({error_units})')

    #If only the error plot is called, axes is not subscriptable
    if number_of_plots == 1:
        fig, axes = plt.subplots(number_of_plots, 1, figsize=(14, 7*number_of_plots)) #Single-pane plots seem to be much smaller than they should be, if using a universal scaling size
        pos = axes.imshow((model_output.squeeze() - target.squeeze()), cmap='bwr', origin='lower', vmin=-1*avg, vmax=avg)
        axes.axis("off")
        cbar = fig.colorbar(pos, ax=axes, fraction=0.021, pad=0.01)
        cbar.set_label(f'Error ({error_units})')
        plt.title(f"{title} \n Date = {date_str} \n Maximum = {maxtemp:.1f} | Minimum = {mintemp:.1f} \n Prediction - Truth (RMSE = {np.sqrt(np.nanmean((model_output.squeeze() - target.squeeze())**2)):.3f})", 
                  va="bottom", fontsize=14) #suptitle also looks bad
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f"{save_dir}/{fig_savename}",dpi=300, bbox_inches="tight")
    
    plt.show()
    
    return

########################################################

def plot_model_vs_model_error(model_1_output, 
                              model_2_output, 
                              pred, 
                              targ, 
                              date_str, 
                              error_units, 
                              title_str=None, 
                              avg_denom=10
                             ):
    
    """
    Plots the difference in absolute errors between 2 models, or a model and Smartinit

    Inputs:
        - model_1_output --> array of model 1's output for whatever variable 
        - model_2_output --> same but for model 2. Can also be Smartinit data
        - pred, targ --> predictor and target data (HRRR and URMA respectively) 
        - date_str --> should be dt_current from get_model_output_at_idx
        - error_units --> string of the format f"{C.varname_units_dict[TARG_VAR]} (+ = [model 2]/[Smartinit] is better)"
        - title_str --> string describing the models, or model + smartinit. Should include a line break (\n) with {dt_current} in it
        - avg_denom --> int, same usage as other plotting functions, to control the scale of the colorbar
    """

    pred, model_1_output, model_2_output, targ = crop_to_intersection_of_regions(pred, model_1_output, model_2_output, targ)

    maxtemp = np.max([np.nanmax(model_1_output.squeeze()), np.nanmax(model_2_output.squeeze()), np.nanmax(targ.squeeze()), np.nanmax(pred.squeeze())])
    mintemp = np.min([np.nanmin(model_1_output.squeeze()), np.nanmin(model_2_output.squeeze()), np.nanmin(targ.squeeze()), np.nanmin(pred.squeeze())])
    avg = (maxtemp-mintemp)/avg_denom
    
    fig, axs = plt.subplots(1,1, figsize=(12,16))

    # Need to plot difference in ABSOLUTE errors, otherwise there's issues with negative regions
    pos = axs.imshow((np.abs(model_1_output.squeeze()-targ.squeeze()) - (np.abs(model_2_output.squeeze()-targ.squeeze()))), cmap="bwr", origin='lower', vmin = -1*avg, vmax = avg)
    axs.axis("off")
    cbar = fig.colorbar(pos, fraction=0.022, pad=0.01)
    cbar.set_label(f"Difference in {error_units}")
    
    if title_str is None:
        title_str = f"Model 1 error minus Model 2 error \n {date_str}"
    
    plt.title(title_str)

    return

########################################################

def plot_model_vs_smartinit_RMSE(model_attrs, 
                                 statsobj_model, 
                                 statsobj_smartinit, 
                                 units_str="UNITS", 
                                 save_fig=False, 
                                 PLOT_SAVE_DIR=os.getcwd()
                                ):
    """
    Inputs: 
        - model_attrs --> instance of DefineModelAttributes class for the current model
        - statsobj_model --> instance of StatObjectConstructor for the current model, with .calc_domain_avg_RMSE_alltimes() already done
        - statsobj_smartinit --> same but for Smartinit
        - units_str --> string for the current variable's units (usually is f"{C.varname_units_dict[TARG_VAR]}")
        - save_fig --> bool to save fig or not
        - PLOT_SAVE_DIR --> full directory path of where to save plots if save_fig=True. Default = current working directory
    """
    
    rmse_diff = np.array(statsobj_smartinit.domain_avg_rmse_alltimes_list) - np.array(statsobj_model.domain_avg_rmse_alltimes_list)

    window_len = 24

    fig, axes = plt.subplots(figsize=(14,7))
    plt.plot(model_attrs.dataset_date_list, rmse_diff, 
             ".", linestyle='None', markersize=0.5, color='g', alpha=0.5, label="RMSE diff.")
    
    plt.plot(model_attrs.dataset_date_list[window_len-1:], rolling_avg(rmse_diff, window_len), 
             linewidth=1, color="g", label=f"RMSE diff., rolling {window_len}-hr mean")
    
    plt.hlines(np.mean(rmse_diff), xmin=model_attrs.dataset_date_list[0], xmax=model_attrs.dataset_date_list[-1], 
               color="r", linewidth=2, label=f"RMSE diff. 2024 mean ({np.mean(rmse_diff):.3f})")
    
    plt.hlines(0, xmin=model_attrs.dataset_date_list[0], xmax=model_attrs.dataset_date_list[-1], linestyle='--', color="k", linewidth=2)
    
    plt.xlim([model_attrs.dataset_date_list[0], model_attrs.dataset_date_list[-1]])

    
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    for label in axes.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    plt.legend(loc="upper left")
    
    plt.ylabel(f"RMSE improvement ({units_str})")
    plt.xlabel("Date")
    plt.title(f"{statsobj_smartinit.target_var} domain-average RMSE difference, Smartinit minus Model, 2024 \n \
                Model = {model_attrs.savename}", fontsize=9)

    if save_fig:
        fig_savename = f"RMSE_{statsobj_smartinit.target_var}_model({model_attrs.savename})"
        plt.savefig(f"{PLOT_SAVE_DIR}/{fig_savename}.png",dpi=300, bbox_inches="tight")
        print(f"{fig_savename} saved to {PLOT_SAVE_DIR}")
        
    plt.show()

    return

########################################################

def plot_model_vs_model_RMSE(model_1_attrs, 
                             model_2_attrs, 
                             statsobj_model_1, 
                             statsobj_model_2, 
                             TARG_VAR, 
                             units_str="UNITS", 
                             save_fig=False, 
                             PLOT_SAVE_DIR=os.getcwd()
                            ):
    """
    Inputs: 
        - model_1_attrs --> instance of DefineModelAttributes class for the first (baseline) model
        - model_2_attrs --> instance of DefineModelAttributes class for the second (comparison) model
        - statsobj_model_1 --> instance of StatObjectConstructor for the first model, with .calc_domain_avg_RMSE_alltimes() already done
        - statsobj_model_2 --> same but for second model
        - TARG_VAR --> string of the desired target variable, e.g. 't2m'
        - units_str --> string for the current variable's units (usually is f"{C.varname_units_dict[TARG_VAR]}")
        - save_fig --> bool to save fig or not
        - PLOT_SAVE_DIR --> full directory path of where to save plots if save_fig=True. SHOULD BE CHANGED FROM DEFAULT BY CALLING FUNCTION!
    """

    rmse_diff = np.array(statsobj_model_1.domain_avg_rmse_alltimes_list) - np.array(statsobj_model_2.domain_avg_rmse_alltimes_list)

    window_len = 24

    fig, axes = plt.subplots(figsize=(14,7))
    plt.plot(model_1_attrs.dataset_date_list, rmse_diff, 
             ".", linestyle='None', markersize=0.5, color='b', alpha=0.5, label="RMSE diff.") 
    
    plt.plot(model_1_attrs.dataset_date_list[window_len-1:], rolling_avg(rmse_diff, window_len), 
             linewidth=1, color="b", label=f"RMSE diff., rolling {window_len}-hr mean")
    
    plt.hlines(np.mean(rmse_diff), xmin=model_1_attrs.dataset_date_list[0], xmax=model_1_attrs.dataset_date_list[-1], 
               color="r", linewidth=2, label=f"RMSE diff. 2024 mean ({np.mean(rmse_diff):.3f})")
    
    plt.hlines(0, xmin=model_1_attrs.dataset_date_list[0], xmax=model_1_attrs.dataset_date_list[-1], linestyle='--', color="k", linewidth=2)
    
    plt.xlim([model_1_attrs.dataset_date_list[0], model_1_attrs.dataset_date_list[-1]])

    
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    for label in axes.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    plt.legend(loc="upper left")
    
    plt.ylabel(f"RMSE improvement ({units_str})")
    plt.xlabel("Date")
    plt.title(f"{TARG_VAR} domain-average RMSE difference, Model 1 minus Model 2, 2024 \n \
                Model 1 = {model_1_attrs.savename} \n \
                Model 2 = {model_2_attrs.savename}", fontsize=9)

    if save_fig:
        fig_savename = f"RMSE_{TARG_VAR}_model1({model_1_attrs.savename})_model2({model_2_attrs.savename})"
        plt.savefig(f"{PLOT_SAVE_DIR}/{fig_savename}.png",dpi=300, bbox_inches="tight")
        print(f"{fig_savename} saved to {PLOT_SAVE_DIR}")
        
    plt.show()

    return

########################################################

def plot_training_loss(TRAINING_LOG_FILEPATH, title_str="", epoch_offset=0, window_len=10):
    """
    Plot epoch loss as a function of epoch number, given a either a full filepath to a training log txt file, or just a training log filename (in which case it will look for the file in C.DIR_UNET_MAIN/Training_logs)
        !!! LINES MUST BE FORMATTED AS SOMETHING LIKE "End of epoch [number] | Average epoch loss = [float] | [whatever]"

    - title_str should be something like current_model_attrs.savename, or whatever; should be descriptive of the model trained
    - epoch_offset --> int to offset the x axis; useful if a model was trained from a checkpoint
    - window_len --> rolling average window length; default = 10 
    """

    C = CONSTANTS()
    
    if "/" not in TRAINING_LOG_FILEPATH: #bad hack but w/e
        TRAINING_LOG_FILEPATH = f"{C.DIR_UNET_MAIN}/Training_logs/{TRAINING_LOG_FILEPATH}"
    
    epoch_number_arr, epoch_loss_arr = read_epoch_num_and_loss(TRAINING_LOG_FILEPATH)
    
    x_arr = [n+epoch_offset for n in epoch_number_arr]
    
    plt.subplots(1,1, figsize=(12,8))
    plt.scatter(x_arr, epoch_loss_arr, s=3)
    plt.plot(x_arr[window_len-1:], rolling_avg(epoch_loss_arr, window_len), color='r')
    
    plt.legend(["Loss", f"{window_len}-epoch average"])
    plt.xlabel("Epoch number")
    plt.ylabel("Epoch loss")
    
    plt.title(f"Training loss, {title_str}")

    return