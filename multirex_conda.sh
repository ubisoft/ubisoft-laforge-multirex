#!/usr/bin/env bash

export CONDA_ENV_NAME=multirex
echo $CONDA_ENV_NAME

conda create -y -n $CONDA_ENV_NAME python=3.8.19

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt

pip install .

# Download FLAME assets from smirk repository
mkdir -p ./assets/FLAME
wget -P ./assets/FLAME https://github.com/georgeretsi/smirk/raw/main/assets/l_eyelid.npy
wget -P  ./assets/FLAME  https://github.com/georgeretsi/smirk/raw/main/assets/landmark_embedding.npy
wget -P  ./assets/FLAME  https://github.com/georgeretsi/smirk/raw/main/assets/r_eyelid.npy
wget -P  ./assets/FLAME https://github.com/georgeretsi/smirk/raw/main/assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz

# Download FLAME and lbs scripts from smirk repository
wget -P ./src/multirex/FLAME https://github.com/georgeretsi/smirk/raw/main/src/FLAME/FLAME.py
wget -P ./src/multirex/FLAME https://github.com/georgeretsi/smirk/raw/main/src/FLAME/lbs.py

# Patch FLAME and lbs scripts for accepting neutral mesh as shape
patch ./src/multirex/FLAME/FLAME.py <  ./assets/FLAME/flame.patch
patch ./src/multirex/FLAME/lbs.py <  ./assets/FLAME/lbs.patch
