#!/bin/bash

#
#source ../../../../venv/bin/activate


source ../../../venv/bin/activate

# log
log_root='./logs'
mkdir -p ${log_root}
date=`date +%Y%m%d_%H%M`
log_file=${log_root}/${date}.log

#
SQUAD_VER=v1.1
#SQUAD_VER=v2.0

#
BERT_BASE_ROOT=~/Projects/08_BERT_MODELS
BERT_BASE_DIR=${BERT_BASE_ROOT}/uncased_L-12_H-768_A-12


#
if [ ${SQUAD_VER} = 'v1.1' ]
then
    echo 'SQuAD1'
    SQUAD_DIR=./SQuAD1
    TRAIN_FILE=${SQUAD_DIR}/train-v1.1.json
    PREDICT_FILE=${SQUAD_DIR}/dev-v1.1.json
    VERSION_2_W_NEG=False
else
    echo 'SQuAD2'
    SQUAD_DIR=./SQuAD2
    TRAIN_FILE=${SQUAD_DIR}/train-v2.0.json
    PREDICT_FILE=${SQUAD_DIR}/dev-v2.0.json
    VERSION_2_W_NEG=True
fi

OUTPUT_DIR=${SQUAD_DIR}

#
MAX_SEQ_LEN=384

#
python ../data/create_finetuning_data.py \
    --squad_data_file=${TRAIN_FILE} \
    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
    --train_data_output_path=${OUTPUT_DIR}/squad_${SQUAD_VER}_train.tf_record \
    --meta_data_file_path=${OUTPUT_DIR}/squad_${SQUAD_VER}_meta_data \
    --fine_tuning_task_type=squad \
    --max_seq_length=${MAX_SEQ_LEN}

