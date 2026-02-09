#!/bin/bash

########################################################################
# Makes the predictor (HRRR) datasets needed for model training/testing. 
# HRRR data for the desired period must be put into the HRRR_RAW_DATA_DIR filepath, in folders with names in YYYYMMDD format corresponding to a date (e.g. '20210101')
#     Keep the filename scheme from the public AWS bucket: hrrr.t[00-23]z.wrfnatf01.grib2
#     Default training set: 1-hr forecast data, hourly from 00:00 UTC 2021-01-01 to 23:00 UTC 2023-12-31
#     Default testing set: 1-hour forecast data, hourly from 00:00 UTC 2024-01-01 to 23:00 UTC 2024-12-31
#       !!! EXTREMELY IMPORTANT: these are VALID times, i.e. one hour AFTER the time specified by t[00-23]z, so they align with URMA's zeroth hour analysis !!!
#       !!! This means 23:00 UTC 2020-12-31 needs to be in the training dataset, and 23:00 UTC 2023-12-31 needs to be EXCLUDED from the testing dataset !!!
#     Date limits are set by what data is downloaded, so only download what you need
# Requires the wgrib2 package! https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/
########################################################################

########################################################################
### Variable and filepath definitions

# The location this script is launched from will serve as the master directory - change this if another master path is desired
MASTER_FILEPATH=$PWD
echo "Master filepath: ${MASTER_FILEPATH}"

# Filepath for raw HRRR data
HRRR_RAW_DATA_DIR="/data1/ai-datadepot/models/hrrr/conus/grib2/"

#Filepath for regridded HRRR data. Will have as many subdirectories as variables (in VAR_LIST)
HRRR_REGRIDDED_DATA_DIR="${MASTER_FILEPATH}/Regridded_HRRR"

#Filepath for the completed HRRR training/testing datasets
HRRR_TRAIN_TEST_DIR="${MASTER_FILEPATH}/HRRR_train_test_datasets"

# Array of varnames, used for filepaths and file names
declare -a VAR_LIST=("pressurf" "t2m" "d2m" "spfh2m" "u10m" "v10m") 

# Array of variable selection strings to use with wgrib2. MUST BE IN THE SAME ORDER AS VAR_LIST!
declare -a VAR_SELECTION_LIST=("PRES:surface" "TMP:2 m" "DPT:2 m" "SPFH:2 m" "UGRD:10 m" "VGRD:10 m") 

########################################################################
### Make all necessary directories (if they don't exist)

mkdir -p "${HRRR_REGRIDDED_DATA_DIR}"
mkdir -p "${HRRR_TRAIN_TEST_DIR}"

for IDX in "${!VAR_LIST[@]}"; do
    mkdir -p "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}"
done

########################################################################
### Make all regridded files

#Go through all raw HRRR files, extract the necessary variable and regrid it 

cd ${HRRR_RAW_DATA_DIR}

for YYYYMMDD in *; do
    cd ${YYYYMMDD}
    for file in *; do
        if [[ ! "${file}" == *"idx"  &&  "${file}" == *"wrfnatf01"* ]]; then #make sure we avoid the grib2 index files, if they exist, AND we only work with the wrfnat 1-hr forecast files (if multiple forecast hours are in the same directory)
            # Extract just the tXXz part of file name, assuming it looks like [something].tXXz.[something].grib2
            tXXz=${file#*.}
            tXXz=${tXXz%.*}
            tXXz=${tXXz%.*}
            
            for IDX in "${!VAR_LIST[@]}"; do
                target_filename="hrrr_${VAR_LIST[$IDX]}_${YYYYMMDD}_${tXXz}_regridded.grib2"

                if [[ ! -e "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}" ]]; then #file DNE in target directory, safe to proceed
                    if [[ -e "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" ]]; then
                        rm "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                    fi
                    wgrib2 ${file} -match "${VAR_SELECTION_LIST[$IDX]}" -grib "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" #subset the appropriate variable into a temporary holding file
                    wgrib2 "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" -set_radius 1:6370000 -set_grib_type c3 -set_bitmap 0 -new_grid_winds grid -new_grid_vectors none -new_grid_interpolation bilinear -new_grid lambert:265.0:25.0:25.0 233.723448:2345:2539.703 19.228976:1597:2539.703 "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                    target_filesize=$(stat -c '%s' "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}")
                    while (( target_filesize < 10000 )); do #check if the output file is tiny (<10 kb); if so then something went wrong, and we redo the process. Rarely happens, but does happen and is extremely annoying to deal with, so a check now saves a lot of pain
                        rm "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                        rm "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                        wgrib2 ${file} -match "${VAR_SELECTION_LIST[$IDX]}" -grib "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                        wgrib2 "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" -set_radius 1:6370000 -set_grib_type c3 -set_bitmap 0 -new_grid_winds grid -new_grid_vectors none -new_grid_interpolation bilinear -new_grid lambert:265.0:25.0:25.0 233.723448:2345:2539.703 19.228976:1597:2539.703 "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                        target_filesize=$(stat -c '%s' "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}")
                    done #end filesize check and while loop
                    echo "${target_filename} DONE"

                else #if file exists, make sure it's not tiny, which happens often if process is interrupted mid-run
                    target_filesize=$(stat -c '%s' "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}")
                    while (( target_filesize < 10000 )); do 
                        echo "${target_filename} | filesize = ${target_filesize} | REMAKING!!"
                        if [[ -e "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" ]]; then
                            rm "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                        fi
                        rm "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                        wgrib2 ${file} -match "${VAR_SELECTION_LIST[$IDX]}" -grib "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                        wgrib2 "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" -set_radius 1:6370000 -set_grib_type c3 -set_bitmap 0 -new_grid_winds grid -new_grid_vectors none -new_grid_interpolation bilinear -new_grid lambert:265.0:25.0:25.0 233.723448:2345:2539.703 19.228976:1597:2539.703 "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                        target_filesize=$(stat -c '%s' "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}")
                    done #end filesize check and while loop
                fi #end target_filename check
                
                if [[ -e "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" ]]; then
                    rm "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                fi

                #Special check: remove the final time (23:00 UTC 2024-12-31) so that it isn't in the testing dataset (as its valid time is 00:00 UTC 2025-01-01)
                if [ "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}" == *"20241231_t23z"* ]; then
                    rm "${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                fi
            done #end "for IDX in "${!VAR_LIST[@]}" "
        fi #end 'if [ ( ! "${file}" == *"idx" ) && ( "${file}" == *"wrfnatf01"* ) ]'
    done #end 'for file in *'
    cd ${HRRR_RAW_DATA_DIR} #return to top and go to the next day
done #end 'for YYYYMMDD in *'

########################################################################
### Concatenate into training and testing datasets 

for IDX in "${!VAR_LIST[@]}"; do
    cd ${HRRR_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}

    #Training set shouldn't have 2023-12-31 23z in it
    echo "Making HRRR training dataset for ${VAR_LIST[$IDX]} (2021/22/23)..."
    output_filename=train_hrrr_alltimes_CONUS_${VAR_LIST[$IDX]}.grib2
    if [[ ! -e ${HRRR_TRAIN_TEST_DIR}/${output_filename} ]]; then
        find . -maxdepth 1 -type f ! -name *'20231231_t23z'* ! -name *'_2024'* -exec cat {} + > ${HRRR_TRAIN_TEST_DIR}/${output_filename}
        echo "HRRR training dataset for ${VAR_LIST[$IDX]} is done"
    else
        echo "!! HRRR training dataset for ${VAR_LIST[$IDX]} already exists at ${HRRR_TRAIN_TEST_DIR}/${output_filename}"
    fi

    #Testing set SHOULD have 2023-12-31 23z in it
    echo "Making HRRR testing dataset for ${VAR_LIST[$IDX]} (2024)..."
    output_filename=test_hrrr_alltimes_CONUS_${VAR_LIST[$IDX]}.grib2
    if [[ ! -e ${HRRR_TRAIN_TEST_DIR}/${output_filename} ]]; then
        cat *_20231231_t23z*.grib2 *_2024*.grib2 > ${HRRR_TRAIN_TEST_DIR}/${output_filename}
        echo "HRRR testing dataset for ${VAR_LIST[$IDX]} is done"
    else
        echo "!! HRRR testing dataset for ${VAR_LIST[$IDX]} already exists at ${HRRR_TRAIN_TEST_DIR}/${output_filename}"
    fi
    
done #end loop over VAR_LIST
