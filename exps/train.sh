#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=8
NODE_RANK=0
BATCH_SIZE=64
ACCUM_STEP=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DATESTR=$(date +"%m-%d-%H-%M")

SAVE_PATH=/home/sethih1/MolNexTR/exps/logs/

mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    main.py \
    --data_path /scratch/phys/sin/sethih1/molnext_data_file \
    --train_file /scratch/phys/sin/sethih1/molnext_data_file/train_pubchem.csv \
    --aux_file /scratch/phys/sin/sethih1/molnext_data_file/uspto_mol/train_uspto.csv --coords_file aux_file \
    --valid_file /scratch/phys/sin/sethih1/molnext_data_file/real/acs.csv \
    --test_file /scratch/phys/sin/sethih1/molnext_data_file/real/acs.csv \
    --vocab_file MolNexTR/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --dynamic_indigo --augment --mol_augment \
    --include_condensed \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --encoder_lr 4e-4 \
    --decoder_lr 4e-4 \
    --save_path $SAVE_PATH --save_mode all \
    --label_smoothing 0.1 \
    --epochs 20 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.02 \
    --print_freq 200 \
    --do_train --do_valid --do_test \
    --fp16 --backend gloo 2>&1

