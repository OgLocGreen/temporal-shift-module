#!/bin/bash

num_splits=$(ls quantize_results | grep ^quantize_results_.* | wc -l)

#ZCU104_arch="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json"
ZCU104_arch="../zcu104_arch/arch.json"
ULTRA96V2_arch="../ultra96v2_arch/arch.json"

ZCU104_out="zcu104/compile_results"
ULTRA96V2_out="ultra96v2/compile_results"

echo "Compiling $num_splits splits..."
units_per_block=(1 3 4 6 3)
num_blocks=4
for ((i=0;i<=$num_blocks;i++)); do
    for ((j=1;j<=${units_per_block[$i]};j++)) do
        printf "\n================ Compiling split # ($i, $j) ====================\n"

        tee_append=""
        if [[ $i -ne 0 ]]; then
            tee_append="-a"
        fi
        options="--options {'save_kernel':'','dump':'graph','mode':'debug'}"
        options="--options {'save_kernel':'','dump':'graph','split_io_mem':'','mode':'debug'}"
        if [[ $i -eq 0 ]]; then
            options="--options {'save_kernel':'','dump':'graph','mode':'debug'}"
        fi

        vai_c_tensorflow --arch "$ZCU104_arch" \
            --frozen_pb "quantize_results/quantize_results_${i}_${j}/deploy_model.pb" \
            --output_dir "$ZCU104_out/compile_results_${i}_${j}" \
            --net_name "tsm_resnet50_8f_${i}_${j}" \
            ${options} \
            2>&1 | tee $tee_append compile_log.txt
    done
done
