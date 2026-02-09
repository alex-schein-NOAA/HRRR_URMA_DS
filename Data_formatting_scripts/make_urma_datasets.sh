#!/bin/bash

########################################################################
# Makes the target (URMA) datasets needed for model training/testing. 
# URMA data for the desired period must be put into the URMA_RAW_DATA_DIR filepath, in folders with names in YYYYMMDD format corresponding to a date (e.g. '20210101')
#     Keep the filename scheme from the public AWS bucket: urma2p5.t00z.2dvaranl_ndfd.grb2_wexp
#     Default training set: zeroth-hour analysis, hourly from 00:00 UTC 2021-01-01 to 23:00 UTC 2023-12-31
#     Default testing set: zeroth-hour analysis, hourly from 00:00 UTC 2024-01-01 to 23:00 UTC 2024-12-31
#     Date limits are set by what data is downloaded, so only download what you need
# Requires the wgrib2 package! https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/
########################################################################

########################################################################
### Variable and filepath definitions

# The location this script is launched from will serve as the master directory - change this if another master path is desired
MASTER_FILEPATH=$PWD
echo "Master filepath: ${MASTER_FILEPATH}"

# Filepath for raw URMA data
URMA_RAW_DATA_DIR="/data1/ai-datadepot/models/urma/2p5km/grib2"

#Filepath for regridded URMA data. Will have as many subdirectories as variables (in VAR_LIST)
URMA_REGRIDDED_DATA_DIR="${MASTER_FILEPATH}/Regridded_URMA"

#Filepath for the completed URMA training/testing datasets
URMA_TRAIN_TEST_DIR="${MASTER_FILEPATH}/URMA_train_test_datasets"

# Array of varnames, used for filepaths and file names
declare -a VAR_LIST=("pressurf" "t2m" "d2m" "spfh2m" "u10m" "v10m") 

# Array of variable selection strings to use with wgrib2. MUST BE IN THE SAME ORDER AS VAR_LIST!
declare -a VAR_SELECTION_LIST=("PRES:surface" "TMP:2 m" "DPT:2 m" "SPFH:2 m" "UGRD:10 m" "VGRD:10 m") 

########################################################################
### Make all necessary directories (if they don't exist)

mkdir -p "${URMA_REGRIDDED_DATA_DIR}"
mkdir -p "${URMA_TRAIN_TEST_DIR}"

for IDX in "${!VAR_LIST[@]}"; do
    mkdir -p "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}"
done

########################################################################
### Make all regridded files

#Go through all raw URMA files, extract the necessary variable and regrid it 

cd ${URMA_RAW_DATA_DIR}

for YYYYMMDD in *; do
    cd ${YYYYMMDD}
    for file in *; do
        if [[ ! "${file}" == *"idx" ]]; then #make sure we avoid the grib2 index files, if they exist
            # Extract just the tXXz part of file name, assuming it looks like [something].tXXz.[something].grib2
            tXXz=${file#*.}
            tXXz=${tXXz%.*}
            tXXz=${tXXz%.*}
            
            for IDX in "${!VAR_LIST[@]}"; do
                target_filename="urma_${VAR_LIST[$IDX]}_${YYYYMMDD}_${tXXz}_regridded.grib2"

                if [[ ! -e "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}" ]]; then #file DNE in target directory, safe to proceed
                    wgrib2 ${file} -match "${VAR_SELECTION_LIST[$IDX]}" -grib "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" #subset the appropriate variable into a temporary holding file
                    wgrib2 "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" -set_radius 1:6370000 -grib_out "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                    target_filesize=$(stat -c '%s' "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}")
                    while (( target_filesize < 10000 )); do #check if the output file is tiny (<10 kb); if so then something went wrong, and we redo the process. Rarely happens, but does happen and is extremely annoying to deal with, so a check now saves a lot of pain
                        rm "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                        rm "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                        wgrib2 ${file} -match "${VAR_SELECTION_LIST[$IDX]}" -grib "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                        wgrib2 "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" -set_radius 1:6370000 -grib_out "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                        target_filesize=$(stat -c '%s' "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}")
                    done #end filesize check and while loop
                    echo "${target_filename} DONE"

                else #if file exists, make sure it's not tiny, which happens often if process is interrupted mid-run
                    target_filesize=$(stat -c '%s' "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}")
                    while (( target_filesize < 10000 )); do 
                        echo "${target_filename} | filesize = ${target_filesize} | REMAKING!!"
                        if [[ -e "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" ]]; then
                            rm "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                        fi
                        rm "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                        wgrib2 ${file} -match "${VAR_SELECTION_LIST[$IDX]}" -grib "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                        wgrib2 "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" -set_radius 1:6370000 -grib_out "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}"
                        target_filesize=$(stat -c '%s' "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/${target_filename}")
                    done #end filesize check and while loop
                fi #end target_filename check

                if [[ -e "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2" ]]; then
                    rm "${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}/TEMP.grib2"
                fi
                
            done #end "for IDX in "${!VAR_LIST[@]}" "
        fi #end 'if [ ( ! "${file}" == *"idx" ) && ( "${file}" == *"wrfnatf01"* ) ]'
    done #end 'for file in *'
    cd ${URMA_RAW_DATA_DIR} #return to top and go to the next day
done #end 'for YYYYMMDD in *'

########################################################################
### Concatenate into training and testing datasets 

for IDX in "${!VAR_LIST[@]}"; do
    cd ${URMA_REGRIDDED_DATA_DIR}/${VAR_LIST[$IDX]}

    echo "Making URMA training dataset for ${VAR_LIST[$IDX]} (2021/22/23)..."
    output_filename=train_urma_alltimes_CONUS_${VAR_LIST[$IDX]}.grib2
    if [[ ! -e ${URMA_TRAIN_TEST_DIR}/${output_filename} ]]; then
        cat *_2021*.grib2 *_2022*.grib2 *_2023*.grib2 > ${URMA_TRAIN_TEST_DIR}/${output_filename}
        echo "URMA training dataset for ${VAR_LIST[$IDX]} is done"
    else
        echo "!! URMA training dataset for ${VAR_LIST[$IDX]} already exists at ${URMA_TRAIN_TEST_DIR}/${output_filename}"
    fi

    echo "Making URMA testing dataset for ${VAR_LIST[$IDX]} (2024)..."
    output_filename=test_urma_alltimes_CONUS_${VAR_LIST[$IDX]}.grib2
    if [[ ! -e ${URMA_TRAIN_TEST_DIR}/${output_filename} ]]; then
        cat *_2024*.grib2 > ${URMA_TRAIN_TEST_DIR}/${output_filename}
        echo "URMA testing dataset for ${VAR_LIST[$IDX]} is done"
    else
        echo "!! URMA testing dataset for ${VAR_LIST[$IDX]} already exists at ${URMA_TRAIN_TEST_DIR}/${output_filename}"
    fi
    
done #end loop over VAR_LIST
