#!/bin/bash

# CSV file
CSVFILE="energy_consumption_log.csv"

# CSV file header
echo "Run,File,Energy_pkg_Joules,Energy_psys_Joules,Time_elapsed_seconds,Memory_used_kb" > $CSVFILE

# Measurement function
measure_energy() {
    local FILEPATH=$1
    local FILENAME=$(basename $FILEPATH)
    local INTERPRETER=$2

    for i in {1..50}
    do
        # Run command and measure energy, time and memory
        OUTPUT=$(sudo perf stat -e power/energy-pkg/ -e power/energy-psys/ /usr/bin/time -v $INTERPRETER $FILEPATH 2>&1)
        
        # Grep values
        ENERGY_PKG=$(echo "$OUTPUT" | grep "power/energy-pkg/" | awk '{print $1}')
        ENERGY_PSYS=$(echo "$OUTPUT" | grep "power/energy-psys/" | awk '{print $1}')
        TIME_ELAPSED=$(echo "$OUTPUT" | grep "Elapsed (wall clock) time" | awk '{print $8}')
        MEMORY_USED=$(echo "$OUTPUT" | grep "Maximum resident set size" | awk '{print $6}')
        
        # Append to CSV file
        echo "$i,$FILENAME,$ENERGY_PKG,$ENERGY_PSYS,$TIME_ELAPSED,$MEMORY_USED" >> $CSVFILE
    done
}

# Python files
measure_energy "./Python/SVM.py" "./venv/bin/python"
measure_energy "./Python/RF.py" "./venv/bin/python"
measure_energy "./Python/NB.py" "./venv/bin/python"
measure_energy "./Python/LR.py" "./venv/bin/python"
measure_energy "./Python/KNN.py" "./venv/bin/python"
measure_energy "./Python/KMC.py" "./venv/bin/python"

# R files
measure_energy "./R/SVM.R" "Rscript"
measure_energy "./R/RF.R" "Rscript"
measure_energy "./R/NB.R" "Rscript"
measure_energy "./R/LR.R" "Rscript"
measure_energy "./R/KNN.R" "Rscript"
measure_energy "./R/KMC.R" "Rscript"

echo "Completed 50 runs for each file. Logs are saved in $CSVFILE"
