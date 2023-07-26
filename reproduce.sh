#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/msc_ml/reproduce.py 


#optional/ some problems for loading checkpoint created by accelerate
accelerate 
accelerate launch --config_file /root/autodl-tmp/msc_ml/se.yaml  /root/autodl-tmp/msc_ml/reproduce.py