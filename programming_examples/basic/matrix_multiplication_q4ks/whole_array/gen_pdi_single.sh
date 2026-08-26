#!/bin/bash

set -euo pipefail

# File to store logs and generated artifacts.
output_file="pdi_gen.log"
output_folder="generated_pdi_insts"

mkdir -p "$output_folder"

# Edit this list directly. Each entry is "maximum-M K N". The script builds
# every M value from 256 through maximum-M in 256-row increments.
gen_sizes=(
    #test
    # "4096 4096 4096"

    # Phi 4 Mini 3.8B
    # "8192 3072 3072"
    # "8192 3072 1024"
    # "8192 3072 8192"
    # "8192 8192 3072"

    # Qwen 2.5 1.5B
    # "8192 1536 1536"
    # "8192 1536 2048"
    # "8192 1536 9216"
    # "8192 9216 1536"

    # Llama 3 8B
    "8192 4096 4096"
    "8192 4096 1024"
    "8192 4096 14336"
    "8192 14336 4096"

    # Gemma 2 2B
    # "8192 2048 2048"
    # "8192 2048 2560"
    # "8192 2048 16384"
    # "8192 16384 2048"

    # Qwen 2.5 14B
    # "8192 5120 5120"
    # "8192 5120 1024"
    # "8192 5120 13824"
    # "8192 13824 5120"

    # Llama 3 70B
    # "8192 8192 8192"
    # "8192 8192 1024"
    # "8192 8192 28672"
    # "8192 28672 8192"
)

# The generated filename describes the user-visible L3 tensor contract.
# Tile size, columns, compute type, accumulation mode, and cache mode all come
# from the local Makefile and are discovered through print-target-suffix.
DT="bf16_q4k_bf16"

step_val=256
failures=0

for size in "${gen_sizes[@]}"; do
    read -r M K N <<< "$size"
    SUBDIR="${DT}_M"
    size_dir="$output_folder/${M}x${K}x${N}/${SUBDIR}"
    mkdir -p "$size_dir"

    for M_SPLIT in $(seq "$step_val" "$step_val" "$M"); do
        build_suffix="$(make --no-print-directory -s print-target-suffix \
            M="$M_SPLIT" K="$K" N="$N")"
        IFS=_ read -r dimensions tile cols _ <<< "$build_suffix"
        output_suffix="${dimensions}_${tile}_${cols}_${DT}"
        xclbin="build/final_${build_suffix}.xclbin"
        pdi="build/final_${build_suffix}.prj/main.pdi"
        insts="build/insts_${build_suffix}.bin"

        echo "Running with M=${M}, K=${K}, N=${N}, M_SPLIT=${M_SPLIT}"

        # Only dimensions are overridden. The fixed tile and all build modes
        # remain manually configurable in the Makefile. pipefail prevents
        # accepting partial compiler output.
        if ! make M="$M_SPLIT" K="$K" N="$N" 2>&1 | tee -a "$output_file"; then
            echo "Failed M_SPLIT=${M_SPLIT}" | tee -a "$output_file"
            failures=$((failures + 1))
            continue
        fi

        if [[ ! -s "$xclbin" || ! -s "$pdi" || ! -s "$insts" ]]; then
            echo "Missing build artifact for ${build_suffix}" | tee -a "$output_file"
            failures=$((failures + 1))
            continue
        fi

        echo "Generated PDI ${pdi}"
        cp -f -- "$pdi" "$size_dir/final_${output_suffix}.pdi"
        cp -f -- "$insts" "$size_dir/insts_${output_suffix}.bin"
    done
done

if (( failures )); then
    echo "PDI generation failed for ${failures} configuration(s)" | tee -a "$output_file"
    exit 1
fi
