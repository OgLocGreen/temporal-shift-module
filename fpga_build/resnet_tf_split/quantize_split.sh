#!/bin/bash
set -e

### USAGE ###
# ./quantize_split.sh <tensorflow_export_path>
# tensorflow_export_path := Path to which tensorflow dumped the split model & quantization information
# e.g. with default path ./quantize_split.sh ../../resnet50_tf_split_export

export DECENT_DEBUG=3

# Constants
num_splits=17
calib_iter=12

# Path to output directory of TF
base_path="$1"
export CALIB_BASE_PATH="$base_path"

base_options="--calib_iter $calib_iter --input_fn calib_input_split.input_fn"

num_split_dirs=$(ls "$base_path" | wc -l)

if [[ num_splits -ne num_split_dirs ]]; then
    echo -e "Number of outputs split directories from: \n\
'$base_path' ($num_split_dirs) not equal to coded num_splits ($num_splits)"
fi

for D in `find "$base_path" -type d`; do
    num_input_files=$(find "$D" -type f -name "inputs_*" | wc -l)
    if [[ calib_iter -gt num_input_files ]]; then
        echo -e "Specified calibration iterations > available inputs for split $D: $calib_iter > $num_input_files"
    fi
done

if [[ $# -eq 0 ]]; then
    echo "Missing arg: Provide path to base split model dir"
    exit 1
fi

units_per_block=(1 3 4 6 3)
num_blocks=4
for ((i=0;i<=$num_blocks;i++)); do
    for ((j=1;j<=${units_per_block[$i]};j++)) do
        printf "\n================ Quantizing block # $i (unit $j) ====================\n"
        model_dir="$base_path/resnet50_tf_split_${i}_${j}"
        config=$(<"$model_dir/quantize_info.txt")
        export CALIB_BLOCK=$i
        export CALIB_UNIT=$j

        tee_append=""
        if [[ $i -ne 0 ]]; then
            tee_append="-a"
        fi

        vai_q_tensorflow quantize --output_dir "quantize_results/quantize_results_${i}_${j}" $base_options --input_frozen_graph "$model_dir/resnet50_tf_split_${i}_${j}.pb" \
            $(echo $config) 2>&1 | tee $tee_append quantize_log.txt
    done
done

