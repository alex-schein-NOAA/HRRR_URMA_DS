

class CONSTANTS():
    def __init__(self):

        """
        Stores constants/variables that are frequently used across different files and classes.
        
        CONTENTS:
        - varname_translation_dict --> converts from my naming conventions for the variables, to how they're actually named in the .grib2 files. Should be used in xarray variable selection, e.g. "xr_dataset[varname_translation_dict["pressurf"]]"
        - urma_selection_dict --> to be used in xarray call as backend_kwargs for loading an URMA file with multiple levels (e.g. the data in Regridded_URMA). 
            >Note the keys are the actual var names, so should go through varname_translation_dict, e.g. "backend_kwargs=urma_selection_dict[varname_translation_dict["pressurf"]]"
            > Also should be used for Smartinit, which shares the naming conventions of URMA
        - Pseudo-means and pseudo-stddevs for variables over the larger western domain, as they're non-trivial to compute. Should be used for normalization in the Dataset class, and for de-normalization in data fetching functions (for model output).
            > Structured as dictionaries, split on HRRR/URMA and again on means/stddevs. Each dict has 2 levels: first select 'train' or 'test' as needed, then '[var_name]' to get the quantity.
            > "Pseudo" because these aren't true means and stddevs - only values, rounded to 6 sig figs, of means/stddevs of data calculated over smaller regions. The important thing is that they NEARLY normalize the data and are universal (same for predictor and target)
        - Commonly used directory paths
        - Patch size for training. Default = 400. Currently only using a isngle value for height and width
        
        """
        
        self.varname_translation_dict = {"pressurf":"sp",
                                         "t2m":"t2m",
                                         "d2m":"d2m",
                                         "spfh2m":"sh2",
                                         "u10m":"u10",
                                         "v10m":"v10"}
        
        self.urma_var_select_dict = {"sp":{'filter_by_keys':{'typeOfLevel': 'surface'}},
                                     "t2m":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                                     "d2m":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}}, 
                                     "sh2":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':2}},
                                     "u10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}},
                                     "v10":{'filter_by_keys':{'typeOfLevel': 'heightAboveGround','level':10}}}

        # For plot labeling purposes
        self.varname_units_dict = {"pressurf":"Pa",
                                   "t2m":"deg C",
                                   "d2m":"deg C",
                                   "spfh2m":"kg/kg",
                                   "u10m":"m/s",
                                   "v10m":"m/s"}
        
        
        self.hrrr_means_dict = {'train':{'pressurf':88264.4,
                                         't2m':284.451, 
                                         'd2m':273.703, 
                                         'spfh2m':0.00529336, 
                                         'u10m':1.17153, 
                                         'v10m':-0.313557}, 
                                'test':{'pressurf':88250.4,
                                         't2m':284.61, 
                                         'd2m':273.83, 
                                         'spfh2m':0.00530784, 
                                         'u10m':1.19730, 
                                         'v10m':-0.339266} }
        
        self.hrrr_stddevs_dict = {'train':{'pressurf':8493.89,
                                         't2m':11.0624, 
                                         'd2m':9.32916, 
                                         'spfh2m':0.00311317, 
                                         'u10m':3.14697, 
                                         'v10m':3.69828}, 
                                  'test':{'pressurf':8490.22,
                                         't2m':10.9822, 
                                         'd2m':9.14753, 
                                         'spfh2m':0.00306908, 
                                         'u10m':3.14104, 
                                         'v10m':3.71392} }
        
        self.urma_means_dict = {'train':{'pressurf':88264.4,
                                         't2m':284.451, 
                                         'd2m':273.703, 
                                         'spfh2m':0.00529336, 
                                         'u10m':1.17153, 
                                         'v10m':-0.313557}, 
                                'test':{'pressurf':88250.4,
                                         't2m':284.61, 
                                         'd2m':273.83, 
                                         'spfh2m':0.00530784, 
                                         'u10m':1.19730, 
                                         'v10m':-0.339266} }

        self.urma_stddevs_dict = {'train':{'pressurf':8493.89,
                                         't2m':11.0624, 
                                         'd2m':9.32916, 
                                         'spfh2m':0.00311317, 
                                         'u10m':3.14697, 
                                         'v10m':3.69828}, 
                                  'test':{'pressurf':8490.22,
                                         't2m':10.9822, 
                                         'd2m':9.14753, 
                                         'spfh2m':0.00306908, 
                                         'u10m':3.14104, 
                                         'v10m':3.71392} }
        

        ## Common directories 
        self.DIR_TRAIN_TEST = f"/scratch3/BMC/wrfruc/aschein/Train_Test_Files"
        self.DIR_UNET_MAIN = f"/scratch3/BMC/wrfruc/aschein/UNet_final"
        self.DIR_TRAINED_MODELS = f"/scratch3/BMC/wrfruc/aschein/UNet_final/Trained_models"
        self.DIR_SMARTINIT_DATA = f"/scratch3/BMC/wrfruc/aschein/SMARTINIT_STUFF/smartinit_2024/output_files" #regridded Smartinit
        self.DIR_SMARTINIT_DATA_NDFD_GRID = f"/scratch3/BMC/wrfruc/aschein/SMARTINIT_STUFF/smartinit_2024/output_files_NDFDgrid" #old Smartinit on native grid

        ## Indexes for selection of patches for training
        self.PATCH_SIZE=400

        ## Region indices (relative to data that has already been restricted to the intersection of HRRR and Smartinit regions)
        self.region_keyword_indices_dict = {"Colorado":{'RESTR_ORIGIN_LAT_IDX':630,
                                                        'RESTR_ORIGIN_LON_IDX':600,
                                                        'RESTR_PATCH_SIZE_LAT':200,
                                                        'RESTR_PATCH_SIZE_LON':200},
                                            "California":{'RESTR_ORIGIN_LAT_IDX':570,
                                                          'RESTR_ORIGIN_LON_IDX':80,
                                                          'RESTR_PATCH_SIZE_LAT':350,
                                                          'RESTR_PATCH_SIZE_LON':200},
                                            "West":{'RESTR_ORIGIN_LAT_IDX':460,
                                                    'RESTR_ORIGIN_LON_IDX':50,
                                                    'RESTR_PATCH_SIZE_LAT':750,
                                                    'RESTR_PATCH_SIZE_LON':750},
                                            "Appalachia":{'RESTR_ORIGIN_LAT_IDX':480,
                                                          'RESTR_ORIGIN_LON_IDX':1400,
                                                          'RESTR_PATCH_SIZE_LAT':400,
                                                          'RESTR_PATCH_SIZE_LON':330},
                                            "Great Lakes":{'RESTR_ORIGIN_LAT_IDX':810,
                                                           'RESTR_ORIGIN_LON_IDX':1150,
                                                           'RESTR_PATCH_SIZE_LAT':400,
                                                           'RESTR_PATCH_SIZE_LON':600},
                                            "Northeast":{'RESTR_ORIGIN_LAT_IDX':770,
                                                         'RESTR_ORIGIN_LON_IDX':1580,
                                                         'RESTR_PATCH_SIZE_LAT':450,
                                                         'RESTR_PATCH_SIZE_LON':450} }

        