from CONSTANTS import *

from HEADER_utilities import *
from HEADER_torch import *

from DefineModelAttributes import *
from HRRR_URMA_Dataset import * #might not be needed

######################################################################################################################################################

class StatsObject():
    """
    Class to aid in computing statistical quantities for trained models and Smartinit.
    """
    def __init__(self,
                 is_smartinit=False,
                 is_conus=True,
                 current_model_attrs=None,
                 predictor_var=None,
                 target_var=None
                ):
            
        """
        - Input vars:
            - is_smartinit --> bool; if True, then calculate stats relative to Smartinit, otherwise use the model in current_model_attrs.model
            - is_conus --> bool, True is desiring stats over the CONUS region. As of 2025/12/03, False does nothing here, but should eventually be used as the flag to calculate over a specific region, which will need to be put as an input argument
                > !!!! WILL ONLY CALCULATE STATS IN THE REGION OF OVERLAP BETWEEN SMARTINIT AND HRRR !!!! This is NOT the entire HRRR domain !!
                > (2025/12/03) Need to add a region keyword argument to define a region to calculate stats over
            - current_model_attrs --> an instance of the DefineModelAttributes class. Doesn't need to have .model set already, as this class should take care of that
            - predictor_var --> string of the predictor var (e.g. 't2m'). Only used for get_model_output_at_idx call; needs to match at least one of the predictor variables in self.current_model_attrs.predictor_vars, but is otherwise not important, and is not used for Smartinit
            - target_var --> string (just one, not a list of multiple!) of the desired target variable whose quantities will be computed
                - Valid options: "pressurf", "t2m", "d2m", "spfh2m", "u10m", "v10m"
        """

        #########################################

        self.C = CONSTANTS()
        
        self.is_smartinit = is_smartinit
        self.is_conus = is_conus
        self.current_model_attrs = current_model_attrs
        self.predictor_var = predictor_var if current_model_attrs==None else current_model_attrs.predictor_vars[0] 
        self.target_var = target_var

        self.smartinit_directory = self.C.DIR_SMARTINIT_DATA
        self.trained_models_directory = self.C.DIR_TRAINED_MODELS
        
        self.domain_avg_rmse_alltimes_list = None

        self.nan_fill_value = 0 #set as an object property for use externally


        ######################################### FUNCTIONS #########################################

    def calc_domain_avg_RMSE_alltimes(self):
        """ 
        Wrapper function for RMSE calculations for HRRR and Smartinit. 

        TO DO (2025-12-03):
            - make an optional argument to restrict the domain - this function might be needed to calc RMSE over subregions like Colorado. Can use restrict_to_region
        """

        if self.domain_avg_rmse_alltimes_list is None: #Only compute if not already done
            self.domain_avg_rmse_alltimes_list = []
            if self.is_smartinit:
                self.calc_smartinit_domain_avg_RMSE_alltimes()
            else:
                self.calc_model_domain_avg_RMSE_alltimes()

        else:
            print(f"Domain average RMSE list already exists - if this needs to be redone, set [obj].domain_avg_RMSE_alltimes_list = None and rerun this function.")
        
        return

    #########################################
    
    def calc_model_domain_avg_RMSE_alltimes(self):
        """
        Calculates domain average RMSE for the given model (in the input model_attrs).
        Currently (as of 2025/12/03) this is only set up to calculate over the CONUS domain.
        Uses only the overlapping region between Smartinit and the model output domains!

        Uses self.nan_fill_value (default=0), so if this should be another value, change that before calling this function.

        TO DO (2025-12-03):
            - make an optional argument to restrict the domain - this function might be needed to calc RMSE over subregions like Colorado. Can use restrict_to_region
        """

        if self.current_model_attrs.dataset is None:
            self.current_model_attrs.create_dataset()

        if self.current_model_attrs.model is None:
            self.current_model_attrs.set_model_weights(self.current_model_attrs.savename) #if the model savename can't be constructed by .create_savename(), then it needs to be manually set in the calling function before passing it to StatObjectConstructor. Same thing if using an architecture that can't be created by .set_model_architecture() - then manually set .model in the calling function before passing to ConstructStatObject

        if self.is_conus:
            print(f"Calculating RMSE for all times ({self.target_var}, {self.current_model_attrs.savename})")
            print(f"Using NaN fill value = {self.nan_fill_value} (standardized units)")
    
            xr_smartinit = get_smartinit_output_at_idx(i=0, target_var='t2m') #we only care about the mask
            smartinit_data = xr_smartinit.data #need it in memory for speed reasons
            for i in range(len(self.current_model_attrs.dataset.xr_datasets_targ[0])):
                predictor, model_output, target, _ = get_model_output_at_idx(self.current_model_attrs, 
                                                                             self.current_model_attrs.model, 
                                                                             predictor_var=self.predictor_var, 
                                                                             target_var=self.target_var, 
                                                                             idx=i, 
                                                                             is_nan=True, 
                                                                             nan_fill_value=self.nan_fill_value)
                self.domain_avg_rmse_alltimes_list.append(self.calc_domain_avg_RMSE_masked_onetime(predictor, smartinit_data, targ, model_output))
                if i%int(len(self.current_model_attrs.dataset.xr_datasets_targ[0])/100)==0:
                    print(f"{(i/len(self.current_model_attrs.dataset.xr_datasets_targ[0]))*100:.0f}% done")
        else:
            print(f"Regional RMSE not yet implemented")
        
        return

    #########################################
    
    def calc_smartinit_domain_avg_RMSE_alltimes(self):
        """
        Calculates domain average RMSE for Smartinit.
        Currently (as of 2025/12/03) this is only set up to calculate over the CONUS domain.
        Uses only the overlapping region between Smartinit and the HRRR domains!

        TO DO (2025-12-03):
            - make an optional argument to restrict the domain - this function might be needed to calc RMSE over subregions like Colorado. Can use restrict_to_region
        """
        if self.is_conus:
            if os.path.exists(f"{self.C.DIR_UNET_MAIN}/Smartinit_stats/smartinit_RMSE_alltimes_CONUS_{self.target_var}.csv"): #CONUS data should already exist on disk
                print(f"{self.target_var} RMSE data for Smartinit exists on disk (CONUS)")
                with open(f"{self.C.DIR_UNET_MAIN}/Smartinit_stats/smartinit_RMSE_alltimes_CONUS_{self.target_var}.csv", 'r', newline='') as file:
                    reader = csv.reader(file)
                    self.domain_avg_rmse_alltimes_list = [np.float32(x) for x in (list(reader))[0]]
                print(f"{self.target_var} RMSE data has been read in")        
            else:
                print(f"{self.target_var} RMSE data for Smartinit (CONUS) does not exist on disk. Calculating now...")
                xr_urma = xr.open_dataarray(f"{self.C.DIR_TRAIN_TEST}/test_urma_alltimes_CONUS_{self.target_var}.grib2", decode_timedelta=True, engine='cfgrib')
                xr_hrrr = xr.open_dataarray(f"{self.C.DIR_TRAIN_TEST}/test_hrrr_alltimes_CONUS_{self.target_var}.grib2", decode_timedelta=True, engine='cfgrib')
                hrrr_arr = xr_hrrr[0].data #need this in memory as a static mask
                for i, urma_arr in enumerate(xr_urma):
                    xr_smartinit = get_smartinit_output_at_idx(idx=i, target_var=self.target_var)
                    self.domain_avg_rmse_alltimes_list.append(self.calc_domain_avg_RMSE_onetime(hrrr_arr, xr_smartinit.data, urma_arr))
                    if i%int(len(xr_urma)/100)==0:
                        print(f"{(i/len(xr_urma))*100:.0f}% done")
                # Write completed data so this doesn't have to be done again
                with open(f"{self.C.DIR_UNET_MAIN}/Smartinit_stats/smartinit_RMSE_alltimes_CONUS_{self.target_var}.csv", "w", newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(self.domain_avg_rmse_alltimes_list)
                print(f"{self.target_var} csv written to disk")
        else:
            print(f"Regional RMSE not yet implemented")
        
        return

    #########################################
    
    def calc_domain_avg_RMSE_onetime(self, hrrr_arr, smartinit_arr, urma_arr, model_output_arr=None):
        """
        Inputs: (NOTE ALL ARRAYS SHOULD BE ON THE FULL WEXP GRID! NO CROPPING INITIALLY!)
            - hrrr_arr --> HRRR data with NaNs included. Only used for masking, so this can be static. 
            - smartinit_arr --> Smartinit data with NaNs included. Can be static if doing RMSE for model output (only need the Smartinit mask), or @ idx if doing Smartinit calculation
            - urma_arr --> URMA data. Should always be @ idx
            - model_output_arr --> model output @ idx. Not needed if doing Smartinit

        Masks data (either Smartinit or HRRR, depending on if self.is_smartinit=True/False respectively) to the intersection of Smartinit/HRRR regions, then returns the domain average RMSE (float), as compared to urma_arr.
        """
        
        _, model_output_arr, smartinit_arr, urma_arr = crop_to_intersection_of_inputs(hrrr_arr, model_output_arr, smartinit_arr, urma_arr)
        
        if self.is_smartinit:
            diff_arr = smartinit_arr - urma_arr
        else:
            diff_arr = model_output_arr - urma_arr
   
        return np.sqrt(np.nanmean(diff_arr**2))