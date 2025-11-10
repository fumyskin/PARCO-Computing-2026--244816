#!/bin/bash
# let's see where this goes...

# Configuration
# define exactly, for each NUMA block, the cores used 
# based on specifications found in experiment1.c
# export OMP_NUM_THREADS=96   # set manually number of threads
export OMP_PROC_BIND=spread # evenly spread threads on the NUMA cores, still PINNING them and assuring thread affinity 
export OMP_PLACES="\
{0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92},\
{1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81,85,89,93},\
{2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78,82,86,90,94},\
{3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91,95}"


MATRICES=("1138_bus/1138_bus.mtx" "utm5940/utm5940.mtx")
RUNS=12
THREADS=(1 2 4 8 16 32 64 96)  # adjust for your system
EXECUTABLES=("spmv_static")
CHUNK_SIZE=(10 100 1000)

# others to add: "spmv_collapse" "spmv_runtime" "spmv_auto" "spmv_chunked"!!!!!!!!!!!
# "spmv_sequential" "spmv_manual"
# Output file
OUTPUT_FILE="results1.csv"
echo "Executable,Threads,Run,ChunkSize,Time" > "$OUTPUT_FILE"

# Loop for each executable
for MATRIX in "${MATRICES[@]}"; do
    echo "Testing matrix: $MATRIX ..."
    for EXEC in "${EXECUTABLES[@]}"; do
        for T in "${THREADS[@]}"; do
            echo "Running $EXEC with $T threads..."
            export OMP_NUM_THREADS=$T

            for ((i=1; i<=RUNS; i++)); do
                if [ "$EXEC" == "spmv_chunked" ]; then
                    for CHUNK in "${CHUNK_SIZE[@]}"; do
                        output=$(./"$EXEC" "$MATRIX" "$CHUNK" | grep "Elapsed time" | awk '{print $3}')
                        echo "$EXEC,$T,$i,$CHUNK,$output" >> "$OUTPUT_FILE"
                    done
                else
                    output=$(./"$EXEC" "$MATRIX" | grep "Elapsed time" | awk '{print $3}')
                    echo "$EXEC,$T,$i,NA,$output" >> "$OUTPUT_FILE"
                fi
            done
        done
    done
done
