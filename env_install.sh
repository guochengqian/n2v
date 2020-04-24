#!/usr/bin/env bash
# make sure command is : source deepgcn_env_install.sh

# install anaconda3.
# cd ~/
# wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
# bash Anaconda3-2019.07-Linux-x86_64.sh

# module load, uncommet if using local machine
#module purge
module load gcc/7.4.0
module load cuda/10.0.130/

# make sure your annaconda3 is added to bashrc
#source activate
#source ~/.bashrc

conda create --name n2v
conda activate n2v
conda install tensorflow-gpu=1.14 keras=2.2.4 cudatoolkit=10.0 python=3.6.8 numpy=1.16 jupyter
pip install -e .
pip install scikit-image







