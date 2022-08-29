#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2
cmd="python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1
  train.py -data data/kp20k_preprocess
 -vocab data/kp20k_preprocess
 -exp_path data/kp20k_exp
 -model_path data/kp20k_exp
 -model_name dcat_seq"

echo $cmd
eval $cmd
