#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate gradiend-de


base_model="bert-base-german-cased"
mode="gradiend_encoder"

echo "Running all der die pairings"
python train.py pairing=NG_F base_model=${base_model} mode=${mode}
python train.py pairing=N_MF base_model=${base_model} mode=${mode}
python train.py pairing=ND_F base_model=${base_model} mode=${mode}
python train.py pairing=AD_F base_model=${base_model} mode=${mode}
python train.py pairing=GA_F base_model=${base_model} mode=${mode}

echo "Running all die das pairings"
python train.py pairing=N_FN base_model=${base_model} mode=${mode}
python train.py pairing=A_FN base_model=${base_model} mode=${mode}


echo "Running all das dem pairings"
python train.py pairing=ND_N base_model=${base_model} mode=${mode}
python train.py pairing=AD_N base_model=${base_model} mode=${mode}

echo "Running all der dem pairings"
python train.py pairing=ND_M base_model=${base_model} mode=${mode}
python train.py pairing=D_MF base_model=${base_model} mode=${mode}
python train.py pairing=D_FN base_model=${base_model} mode=${mode}

echo "Running all der des pairings"
python train.py pairing=NG_M base_model=${base_model} mode=${mode}
python train.py pairing=G_MF base_model=${base_model} mode=${mode}
python train.py pairing=G_FN base_model=${base_model} mode=${mode}


echo "Running all das des pairings"
python train.py pairing=NG_N base_model=${base_model} mode=${mode}
python train.py pairing=GA_N base_model=${base_model} mode=${mode}
