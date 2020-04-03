#!/bin/bash

source ../../../venv/bin/activate





############################################################
# setup
############################################################

#
SQUAD_VER=v1.1
#SQUAD_VER=v2.0



# mode - enum defined in run_squad_helper.py
mode='train_and_eval'
#mode='train_and_predict'
#mode='train'
#mode='eval'
#mode='predict'
#mode='export_only'


############################################################
# hyperparameters
############################################################

#
BATCH_SIZE_TRAIN=32
BATCH_SIZE_PREDICT=8

#
NUM_TRAIN_EPOCHS=2

#
MAX_SEQ_LEN=384



############################################################
# PATH setup
############################################################

#
OUTPUT_DIR="./output"


# BERT base model
BERT_BASE_ROOT=~/Projects/08_BERT_MODELS_V2
BERT_BASE_DIR=${BERT_BASE_ROOT}/uncased_L-12_H-768_A-12

#
INIT_CKPT=${BERT_BASE_DIR}/bert_model.ckpt

# log
log_root='./logs'

############################################################
# Don't touch
############################################################

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


# log
mkdir -p ${log_root}
date=`date +%Y%m%d_%H%M`
log_file=${log_root}/${date}.log


#
mkdir -p ${OUTPUT_DIR}

#
python run_squad.py \
    --mode=${mode} \
    --input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VER}_meta_data \
    --train_data_path=${SQUAD_DIR}/squad_${SQUAD_VER}_train.tf_record \
    --predict_file=${PREDICT_FILE} \
    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
    --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
    --init_checkpoint=${INIT_CKPT} \
    --train_batch_size=${BATCH_SIZE_TRAIN} \
    --predict_batch_size=${BATCH_SIZE_PREDICT} \
    --learning_rate=8e-5 \
    --num_train_epochs=${NUM_TRAIN_EPOCHS} \
    --model_dir=${OUTPUT_DIR}\
    --distribution_strategy=mirrored

