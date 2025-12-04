from FunctionsAndClasses.CONSTANTS import *

from FunctionsAndClasses.HEADER_torch import *
from FunctionsAndClasses.HEADER_utilities import *
from FunctionsAndClasses.HEADER_models import *

from FunctionsAndClasses.HRRR_URMA_Dataset import *
from FunctionsAndClasses.utils_data import *

#######################################################################################################################

class DefineModelAttributes():
    def __init__(self,
                 is_train=True,
                 is_patches=False,
                 is_attention_model=False,
                 is_residual_model=False,
                 BATCH_SIZE=1,
                 NUM_EPOCHS=1,
                 predictor_vars=["pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"],
                 target_vars=["pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"],
                 with_terrains=["diff"]
                ):
        """
        Class to aid in constructing models and their Pytorch datasets for use in training. MUST set an object of this class whose attributes can then be used to initialized and control model training.

        Input vars:
        - is_train --> bool to determine if training data (2021/22/23) or testing data (2024) is loaded by the HRRR_URMA_Dataset class
        - is_patches --> bool to determine if data is loaded in patches or over the whole CONUS domain. Should be True for training and False for evaluation. Note that patch size is controlled by self.dataset.C.PATCH_SIZE, and generally shouldn't need to be changed from default (400, as of 2025/12/01)
        - is_attention_model --> bool to determine if one of the attention model architectures is used
        - is_residual_model --> bool to determine if one of the residual model architectures is used
        - BATCH_SIZE --> int to determine batch size per GPU for training, or for model selection (if "BS{BATCH_SIZE}" is in the savename)
        - NUM_EPOCHS --> int to determine number of epochs the model will train for, or for model selection (if "NE{NUM_EPOCHS}" is in the savename)
        - predictor_vars --> list of strings of predictor variables to include (e.g. ["t2m", "d2m"]). Default = all variables
        - target vars --> list of strings of target variables to include. Default = all variables
        - with_terrains --> list of strings of terrains to include ("hrrr", "urma", "diff"). Default = ["diff"]
        """

        #########################################

        self.C = CONSTANTS()
        
        self.is_train = is_train
        self.is_patches = is_patches
        self.is_attention_model = is_attention_model
        self.is_residual_model = is_residual_model
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.predictor_vars = predictor_vars
        self.target_vars = target_vars
        self.with_terrains = with_terrains

        ## Make default savename. Can be changed later if model attributes from a saved model are needed
        self.create_savename()

        self.dataset = None #call create_dataset() in whatever calling function needs it
        self.num_channels_in = None
        self.num_channels_out = None

        self.model = None #call set_model_architecture() in whatever calling function needs it

    ######################################### FUNCTIONS #########################################
    
    def create_savename(self):
        """
        Savename is additive, i.e. only includes attributes present in the model
        Ordering: 
            - "residual" if is_residual_model
            - "attn" if is_attention_model
            - Batch size, formatted as BS{self.BATCH_SIZE}
            - Num epochs, formatted as NE{self.NUM_EPOCHS}
            - Terrains (tH, tU, tD for HRRR, URMA, difference terrains respectively)
            - Predictor variables, in parentheses, separated by dashes
                Example: pred(t2m-d2m-u10m)
            - Target variables, in parentheses, separated by dashes
                Example: targ(t2m-pressurf) 
        """
        
        ## Define all optional/constructive arguments
        attn_str = ""
        residual_str = ""
        conus_str = ""
        terrain_str = ""
        
        if self.is_attention_model:
            attn_str = "attn"
        if self.is_residual_model:
            residual_str = "residual"
        terrain_str = "_".join([f"t{x[0].capitalize()}" for x in self.with_terrains]) if self.with_terrains is not None else ""

        # Doesn't play nice if defined within the f-string
        # Making as variables so these can be called independently for other plotting purposes (don't forget to add "pred" and "targ" in calling code!)
        self.predictor_str = "-".join(self.predictor_vars)
        self.target_str = "-".join(self.target_vars)

        savename_attrs_list = [residual_str, 
                               attn_str,
                               f"BS{self.BATCH_SIZE}", 
                               f"NE{self.NUM_EPOCHS}", 
                               f"{terrain_str}", 
                               f"pred({self.predictor_str})", 
                               f"targ({self.target_str})"]
        
        self.savename = "_".join([x for x in savename_attrs_list if x !=""])
        
        return

    #########################################

    def create_dataset(self):
        """ 
        Creates the requisite Dataset for use in Pytorch Dataloader
        NOT called by default due to calculation expense - must be invoked by a calling function 
        """

        # Check if this was already done, to save duplicate computation
        if self.dataset is None:
            print(f"Making dataset for model {self.savename}")

            if is_patches:
                print(f"is_patches = {self.is_patches}; making patches dataset")
            else:
                print(f"is_patches = {self.is_patches}; making CONUS dataset")

            self.dataset = HRRR_URMA_Dataset(is_train = self.is_train,
                                             is_patches = self.is_patches,
                                             predictor_vars = self.predictor_vars,
                                             target_vars = self.target_vars,
                                             with_terrains = self.with_terrains)
            
            # Returns a list of dt.datetime objects of all dates in the current dataset. Useful for plotting and time series stuff
            self.dataset_date_list = [dt.datetime.strptime(str(np.datetime_as_string(date, unit='s')), "%Y-%m-%dT%H:%M:%S") for date in self.dataset.xr_datasets_predictor[0].valid_time.data]
        
        else:
            print(f"Dataset for the model {self.savename} was already computed. If it needs to be recomputed, set [current model].dataset=None and rerun .create_dataset()")
        
        self.num_channels_in = np.shape(self.dataset[0][0])[0]
        self.num_channels_out = np.shape(self.dataset[0][1])[0] #Don't forget to put this in the model definition when targeting >1 var!!!
        return

    #########################################
    
    def set_model_attrs_from_savename(self, savename):
        """
        Input: 
            - savename --> str of model savename, formatted as in self.create_savename() - does NOT need a filepath, only the savename (can include ".pt" extension but generally shouldn't).
        
        Sets the following model attributes:
            - is_attention_model
            - is_residual_model
            - BATCH_SIZE
            - NUM_EPOCHS
            - with_terrains (the whole list)
            - predictor_vars (the whole list)
            - target_vars (the whole list)
        """
        self.with_terrains = [] #reset to an empty list
        strs = savename.split("_")
        for string in strs:
            if "attn" in string:
                self.is_attention_model=True
            elif "residual" in string:
                self.is_residual_model=True
            elif "BS" in string:
                self.BATCH_SIZE = int("".join([char for char in string if char.isdigit()]))
            elif "NE" in string:
                self.NUM_EPOCHS = int("".join([char for char in string if char.isdigit()]))
            elif string=="tH":
                self.with_terrains.append("hrrr")
            elif string=="tU":
                self.with_terrains.append("urma")
            elif string=="tD":
                self.with_terrains.append("diff")
            elif "pred(" in string:
                self.predictor_vars = ((string.split("(")[1])[:-1]).split("-")
            elif "targ(" in string:
                if ".pt" in string: #bad hack but w/e... this whole function is a bad hack
                    self.target_vars = ((string.split("(")[1])[:-4]).split("-")
                else:
                    self.target_vars = ((string.split("(")[1])[:-1]).split("-")
        
        print(f"Model attributes set from {savename}")
        self.create_savename() #needs to be re-called to properly set attributes
        print(f"Savename set to {self.savename}")
        
        return
        
    #########################################

    def set_model_architecture(self):
        """
        Sets model architecture based on parameters.
        Actually makes the model, which is attached to this DefineModelAttributes object.
        
        THIS DOES NOT LOAD WEIGHTS!!!! The specific model weights to use (if any) is still the responsibility of the calling function! This is just to get the right framework. To set weights, use "{obj}.set_model_weights_from_savename(savename)" which in turn will call this function automatically
        """
        
        if self.dataset is None:
            self.create_dataset() #needed for num_channels_{in/out}
        else:
            if self.model is None:
                if self.is_attention_model and not self.is_residual_model:
                    self.model = UNet_Attention_simple(n_channels_in=self.num_channels_in, n_channels_out=self.num_channels_out)
                    print(f"Model architecture set: UNet_Attention_simple")
                elif self.is_residual_model and not self.is_attention_model:
                    self.model = UNet_Residual(n_channels_in=self.num_channels_in, n_channels_out=self.num_channels_out)
                    print(f"Model architecture set: UNet_Residual")
                elif self.is_residual_model and self.is_attention_model:
                    self.model = UNet_Residual_Attention(n_channels_in=self.num_channels_in, n_channels_out=self.num_channels_out)
                    print(f"Model architecture set: UNet_Residual_Attention")
                else: #default to simple UNet
                    self.model = UNet_simple(n_channels_in=self.num_channels_in, n_channels_out=self.num_channels_out)
                    print(f"Model architecture set: UNet_simple")
            else:
                print(f"Model architecture was already set previously! Set ''self.model=None'' and rerun this function if a new model architecture is desired")

        return

    #########################################

    def set_model_weights_from_savename(self, model_savename, is_different_path=False):
        """
            Inputs: 
                - model savename --> string of the model savename whose weights we want to load 
                    > The model name SHOULD NOT include the path, or .pt! (Saved models assumed to be in the Trained_models dir)
                - is_different_path --> bool to load a model from a different directory than the default "Trained_models" dir. Default = False; there are very few cases where this should be set to True
                    > If True, then 'model_savename' should include the ENTIRE path, AND .pt at the end!
            
            This will first set the model attributes from the savename, then create the dataset if not done already, then set the model architecture, and finally load the model weights. The intention is for this to be the only necessary function to call after instantiation in order to get a functioning .model for that object.

            Once model weights are loaded, this updates self.savename to match model_savename, if they don't already match.
        """
        if not is_different_path:
            self.set_model_attrs_from_savename(model_savename)
        else:
            self.set_model_attrs_from_savename(model_savename.split("/")[-1]) #will feed in the model name w/ ".pt" at the end, which is accounted for
        
        if self.dataset is None:
            self.create_dataset()
        
        if self.model is None:
            self.set_model_architecture() #if this should be skipped (e.g. doing a model with an architecture not supported in set_model_architecture), then make sure self.model is manually set in the calling function BEFORE using this one! 
        
        device = torch.device("cuda")
        self.model.to(device)
        if is_different_path:
            self.model.load_state_dict(torch.load(f"{model_savename}", weights_only=True))
        else:
            self.model.load_state_dict(torch.load(f"{self.C.DIR_TRAINED_MODELS}/{model_savename}.pt", weights_only=True))

        print(f"Weights for {model_savename} loaded")

        if (not is_different_path) and (self.savename != model_savename):
            self.savename = model_savename #generally we want this to match the calling savename
            print(f"Savename set to ''{self.savename}'' - rerun .create_savename() if this is not desired")
        
        return