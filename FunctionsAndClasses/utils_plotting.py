from HEADER_torch import *
from HEADER_utilities import *
from HEADER_plotting import *

from CONSTANTS import *

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
    pos = axes[3].imshow((model_output.squeeze() - target.squeeze()), cmap="coolwarm", vmin = -1*avg, vmax = avg, origin='lower') 
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