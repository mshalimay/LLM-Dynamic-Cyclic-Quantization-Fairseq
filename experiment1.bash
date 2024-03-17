#!/bin/bash

ROBERTA_PATH=roberta.base/model.pt
BASE_DIR="/home/mshalimay/fairseq" # ABSOLUTE path to fairseq directory
CUDA_VISIBLE_DEVICES=0 
BASE_SAVE_DIR="/home/mshalimay/fairseq/checkpoints_experiment1"
cpt_cfg_path=$BASE_DIR/fairseq/cfg_cpt.json    # PATH to dynamic quantization parametrization

# datasets hashmap
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

bits=(4)
datasets=(RTE CoLA MRPC STS-B SST-2 QNLI QQP MNLI)

max_epoch=10
freeze=True
quantin=True
quantw=True
measurein=True
measurew=True
fp16=False  # disable's fairseq's quantization

# update cpt_cfg.json
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

    # Create a new cpt_cfg.json with desired parameters
    echo "{
      \"wbit_min\": $wbit_min,
      \"wbit_max\": $wbit_max,
      \"inbit_min\": $inbit_min,
      \"inbit_max\": $inbit_max,
      \"max_epoch\": $max_epoch,
      \"num_cyclic_period\": $num_cyclic_period
    }" > $cpt_cfg_path

    echo "Created new $cpt_cfg_path"
}

for bit in "${bits[@]}"; do
    cpt_config $bit $bit $bit $bit 1
    
    for dataset in "${datasets[@]}"; do
        config=${dataset_config_map[$dataset]}
        DATA_DIR=$BASE_DIR/$dataset
        SAVE_DIR="${BASE_SAVE_DIR}/${dataset}/bits_${bit}_epoch_${max_epoch}"
        
        fairseq-hydra-train \
            --config-dir ./config/finetuning \
            --config-name $config \
            task.data=$DATA_DIR-bin checkpoint.restore_file=$ROBERTA_PATH checkpoint.save_dir=$SAVE_DIR \
            common.fp16=$fp16 \
            +model.freeze=$freeze +model.quantin=$quantin +model.quantw=$quantw\
            +model.measurein=$measurein +model.measurew=$measurew \
            +model.bitsin=$bit +model.bitsw=$bit \
            optimization.max_epoch=$max_epoch
    done
done

