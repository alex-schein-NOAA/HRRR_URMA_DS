#######################################################################################################################################
"""
Example script for training a model across multiple GPU nodes. Set up to train a t2m --> t2m residual attention model with terrain information.
See the referenced scripts for further documentation.
"""
#######################################################################################################################################

from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.CONSTANTS import *
from FunctionsAndClasses.DefineModelAttributes import *
from FunctionsAndClasses.TrainOneModel_DDP import *

C = CONSTANTS()

###############

BATCH_SIZE = 60
NUM_EPOCHS = 150

#######################################################################################################################################

TRAINING_LOG_FILEPATH = f"{C.DIR_UNET_MAIN}/Training_logs/example_training_log_BS{BATCH_SIZE}_NE{NUM_EPOCHS}.txt"

current_model_attrs = DefineModelAttributes(is_train=True,
                                            is_patches=True,
                                            is_attention_model=True,
                                            is_residual_model=True,
                                            with_terrains=None,
                                            predictor_vars=['t2m'],
                                            target_vars=['t2m'],
                                            BATCH_SIZE=BATCH_SIZE,
                                            NUM_EPOCHS=NUM_EPOCHS)

current_model_attrs.create_dataset()
current_model_attrs.set_model_architecture()

TrainOneModel_DDP(current_model_attrs,
                  INITIAL_LEARNING_RATE=2e-5,
                  NUM_WORKERS=10,
                  TRAINING_LOG_FILEPATH = TRAINING_LOG_FILEPATH,
                  TRAINED_MODEL_SAVEPATH = C.DIR_TRAINED_MODELS)
