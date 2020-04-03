#!/bin/bash


source ../../../venv/bin/activate


BERT_BASE_ROOT=~/Projects/08_BERT_MODELS
BERT_BASE_DIR=${BERT_BASE_ROOT}/uncased_L-12_H-768_A-12





# init checkpoint v1
INIT_CKPT=${BERT_BASE_DIR}/bert_model.ckpt

#
BERT_CONFIG=${BERT_BASE_DIR}/bert_config.json

# checkpoint v2
OUTPUT_V2=${BERT_BASE_DIR}/tf2/bert_model_tf2.ckpt


python tf2_encoder_checkpoint_converter.py \
    --bert_config_file=${BERT_CONFIG} \
    --checkpoint_to_convert=${INIT_CKPT} \
    --converted_checkpoint_path=${OUTPUT_V2}
