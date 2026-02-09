Contains all the necessary files for recreating and extending the HRRR-URMA downscaling UNet. 

## Workflow:
1. Download or otherwise have access to hourly HRRR 1-hour forecast files (can be found at https://noaa-hrrr-bdp-pds.s3.amazonaws.com/index.html) containing the variables of interest. The following scripts are set up to deal with any files with `wrfnatf01` in the name.
2. Download or otherwise have access to zeroth-hour URMA analysis files (can be found at https://noaa-urma-pds.s3.amazonaws.com/index.html) containing the variables of interest. Should be any file with `t00z.2dvaranl_ndfd.grb2_wexp` in the name. 
3. Run `create_hrrr_datasets.sh` and `create_urma_datasets.sh`, ensuring your HRRR and URMA repositories are set up appropriately. The scripts default to making training sets from 2021-01-01 00 UTC to 2023-12-31 23 UTC data (valid times) and testing datasets from 2024-01-01 00 UTC to 2024-12-31 23 UTC data.
4. Run `training_multinode_example.py`, modifying options as needed (default setup: 4 nodes with 2 GPUs/node).
5. Examine the output model with `view_results.ipynb`.

Additional information can be found in the file/function documentation. 

The data currently in Model_stats can be overwritten with the stats from your trained model(s); the Smartinit stats are static and would need to be recomputed from Smartinit output, which is beyond the scope of this repository. 
