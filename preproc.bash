#!/bin/bash

# datasets=(CoLA MNLI MRPC QNLI QQP RTE SST-2 STS-B)
datasets=(CoLA MNLI MRPC QNLI QQP SST-2 STS-B)
preprocess_script=./examples/roberta/preprocess_GLUE_tasks.sh  

for dataset in ${datasets[@]}; do
    $preprocess_script glue_data $dataset
done

 