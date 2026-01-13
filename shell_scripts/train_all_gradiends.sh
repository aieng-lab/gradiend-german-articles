#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate gradiend-de

#base_model="EuroBERT/EuroBERT-210m"

base_model='results/decoder-mlm-head-gender-de/AF-AM-AN-DF-DM-DN-GF-GM-GN-NF-NM-NN/german-gpt2'
base_model="bert-base-german-cased"
base_model="LSX-UniWue/ModernGBERT_134M"
base_model="LSX-UniWue/ModernGBERT_1B"

mode="gradiend_decoder"
mode="gradiend_encoder"


echo "Running all gender-case pairings for base model: ${base_model}"

echo "Running all gender pairings"
python train.py pairing=MF base_model=${base_model} mode=${mode}
python train.py pairing=FN base_model=${base_model} mode=${mode}
python train.py pairing=MN base_model=${base_model} mode=${mode}
python train.py pairing=MFN base_model=${base_model} mode=${mode}

echo "Running all case pairings"
python train.py pairing=NG base_model=${base_model} mode=${mode}
python train.py pairing=ND base_model=${base_model} mode=${mode}
python train.py pairing=NA base_model=${base_model} mode=${mode}
python train.py pairing=GD base_model=${base_model} mode=${mode}
python train.py pairing=GA base_model=${base_model} mode=${mode}
python train.py pairing=DA base_model=${base_model} mode=${mode}

echo "Running all der dem pairings"
python train.py pairing=ND_M base_model=${base_model} mode=${mode}
python train.py pairing=D_MF base_model=${base_model} mode=${mode}
python train.py pairing=D_FN base_model=${base_model} mode=${mode}

echo "Running all der des pairings"
python train.py pairing=NG_M base_model=${base_model} mode=${mode}
python train.py pairing=G_MF base_model=${base_model} mode=${mode}
python train.py pairing=G_FN base_model=${base_model} mode=${mode}

echo "Running all der die pairings"
python train.py pairing=N_MF base_model=${base_model} mode=${mode}
python train.py pairing=ND_F base_model=${base_model} mode=${mode}
python train.py pairing=NG_F base_model=${base_model} mode=${mode}
python train.py pairing=AD_F base_model=${base_model} mode=${mode}
python train.py pairing=GA_F base_model=${base_model} mode=${mode}

echo "Running all gender specific case pairings"
python train.py pairing=NG_F base_model=${base_model} mode=${mode}
python train.py pairing=NG_M base_model=${base_model} mode=${mode}
python train.py pairing=NG_N base_model=${base_model} mode=${mode}
python train.py pairing=NA_M base_model=${base_model} mode=${mode}
python train.py pairing=ND_F base_model=${base_model} mode=${mode}
python train.py pairing=ND_M base_model=${base_model} mode=${mode}
python train.py pairing=ND_N base_model=${base_model} mode=${mode}
python train.py pairing=GA_F base_model=${base_model} mode=${mode}
python train.py pairing=GA_M base_model=${base_model} mode=${mode}
python train.py pairing=GA_N base_model=${base_model} mode=${mode}
python train.py pairing=GD_M base_model=${base_model} mode=${mode}
python train.py pairing=GD_N base_model=${base_model} mode=${mode}
python train.py pairing=AD_F base_model=${base_model} mode=${mode}
python train.py pairing=AD_M base_model=${base_model} mode=${mode}
python train.py pairing=AD_N base_model=${base_model} mode=${mode}

echo "Running all case specific gender pairings"
python train.py pairing=N_MF base_model=${base_model} mode=${mode}
python train.py pairing=G_MF base_model=${base_model} mode=${mode}
python train.py pairing=A_MF base_model=${base_model} mode=${mode}
python train.py pairing=D_MF base_model=${base_model} mode=${mode}
python train.py pairing=N_MN base_model=${base_model} mode=${mode}
python train.py pairing=A_MN base_model=${base_model} mode=${mode}
python train.py pairing=N_FN base_model=${base_model} mode=${mode}
python train.py pairing=G_FN base_model=${base_model} mode=${mode}
python train.py pairing=A_FN base_model=${base_model} mode=${mode}
python train.py pairing=D_FN base_model=${base_model} mode=${mode}
