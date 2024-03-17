#!/bin/bash

ROBERTA_PATH=roberta.base/model.pt
BASE_DIR="/home/mshalimay/fairseq"
CUDA_VISIBLE_DEVICES=0 
BASE_SAVE_DIR="/home/mshalimay/fairseq/checkpoints_experiment2"
cpt_cfg_path=$BASE_DIR/fairseq/cfg_cpt.json


max_epoch=5
freeze=False
quantin=True
quantw=True
measurein=False
measurew=False
num_cyclic=64
datasets=(RTE CoLA MRPC STS-B SST-2)
tasks=(1 2 3 4 5 6)

fp16=False

declare -A dataset_config_map=(
    [CoLA]=cola
    [MNLI]=mnli
    [MRPC]=mrpc
    [QNLI]=qnli
    [QQP]=qqp
    [RTE]=rte
    [SST-2]=sst_2
    [STS-B]=sts_b
)

declare -A cpt_config_map=(
    [1]="4 8 16 16 $num_cyclic"  # wbit_min wbit_max inbit_min inbit_max num_cyclic_period
    [2]="4 16 16 16 $num_cyclic"
    [3]="8 16 16 16 $num_cyclic"
    [4]="8 8 16 16 $num_cyclic"
    [5]="16 16 16 16 $num_cyclic"
    [6]="16 16 16 16 $num_cyclic" # dummy task for no quantization
)

# update function for cpt_cfg.json
cpt_config() {
    local wbit_min=$1
    local wbit_max=$2
    local inbit_min=$3
    local inbit_max=$4
    local num_cyclic_period=$5

    # Check if cpt_cfg.json exists, remove it if it does
    if [ -f "$cpt_cfg_path" ]; then
        echo "Removing old $cpt_cfg_path"
        rm "$cpt_cfg_path"
    fi

    # Create a new cpt_cfg.json
    echo "{
      \"wbit_min\": $wbit_min,
      \"wbit_max\": $wbit_max,
      \"inbit_min\": $inbit_min,
      \"inbit_max\": $inbit_max,
      \"num_cyclic_period\": $num_cyclic_period
    }" > $cpt_cfg_path

    echo "Created new $cpt_cfg_path"
}

for task_id in "${tasks[@]}"; do
    config_string=${cpt_config_map[$task_id]}
    IFS=' ' read -r -a config_array <<< "$config_string"
    cpt_config "${config_array[@]}"
    
    bitsin=${config_array[0]}
    bitsw=${config_array[2]}

    for dataset in "${datasets[@]}"; do
        config=${dataset_config_map[$dataset]}
        DATA_DIR=$BASE_DIR/$dataset
        SAVE_DIR="${BASE_SAVE_DIR}/case_${task_id}_epoch_${max_epoch}_fp16_${fp16}/${dataset}/"
        
        fairseq-hydra-train \
            --config-dir ./config/finetuning \
            --config-name $config \
            task.data=$DATA_DIR-bin checkpoint.restore_file=$ROBERTA_PATH checkpoint.save_dir=$SAVE_DIR \
            common.fp16=$fp16 \
            +model.freeze=$freeze +model.quantin=$quantin +model.quantw=$quantw\
            +model.measurein=$measurein +model.measurew=$measurew \
            +model.bitsin=$bitsin +model.bitsw=$bitsw \
            optimization.max_epoch=$max_epoch
    done
done

