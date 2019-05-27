#!/bin/bash

set -x

DATAHOME=${@:(-2):1}
EXEHOME=${@:(-1):1}

SAVEPATH=${DATAHOME}/models/NQG_ner
RESULTPATH=${DATAHOME}/results/NQG_ner

mkdir -p ${SAVEPATH}
mkdir -p ${RESULTPATH}

cd ${EXEHOME}

python run.py \
       -save_path ${SAVEPATH} -log_home ${SAVEPATH} \
       -online_process_data \
       -train_src ${DATAHOME}/train/train.src.txt -src_vocab ${DATAHOME}/train/vocab.src.txt.30k \
       -train_tgt ${DATAHOME}/train/train.tgt.txt -tgt_vocab ${DATAHOME}/train/vocab.tgt.txt.30k \
       -train_ans ${DATAHOME}/train/train.ans.txt -ans_vocab ${DATAHOME}/train/vocab.ans.txt.30k \
       -train_feats ${DATAHOME}/train/train.ner.txt -feat_vocab ${DATAHOME}/train/vocab.ner.txt \
       -layers 1 \
       -enc_rnn_size 384 -brnn -copy -feature -answer encoder \
       -dec_rnn_size 384 -att_vec_size 384 \
       -answer_enc_rnn_size 384 -answer_brnn \
       -word_vec_size 300 \
       -dropout 0.5 \
       -batch_size 32 \
       -beam_size 5 \
       -epochs 20 -optim adam -learning_rate 0.001 \
       -gpus 0 \
       -curriculum 0 -extra_shuffle \
       -start_eval_batch 500 -eval_per_batch 500 -halve_lr_bad_count 3 \
       -seed 12345 -cuda_seed 12345 \
       -log_interval 100 \
       -dev_input_src ${DATAHOME}/dev/dev.src.txt \
       -dev_ref ${DATAHOME}/dev/dev.tgt.txt \
       -dev_ans ${DATAHOME}/dev/dev.ans.txt \
       -dev_feats ${DATAHOME}/dev/dev.ner.txt \
       -paragraph -max_sent_length 272 \
       -result_path ${RESULTPATH}
