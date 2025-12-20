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