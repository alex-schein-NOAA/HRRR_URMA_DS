from HEADER_utilities import *

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