#!/bin/bash

# Copyright 2024 Rev.com  (jp@rev.com)
#
#  This is the main script to launch traing with early 2024 WeNet codebase
# 
extra_train_args=
extra_test_params=
num_workers=8 
prefetch=100
export WANDB_API_KEY=local-473ad2cf1f9ed9023faf837048e75943e1bbe7c5
export WANDB_PORT=30433

ulimit -n 8096

. ./path.sh || exit 1;
. local/functions.sh

#CUDA_VISIBLE_DEVICES=0

# if [ "x$CUDA_VISIBLE_DEVICES" == "x" ]; then
#    echo "We'll detect all available CUDA devices"
#     # Automatically detect number of gpus
#     if command -v nvidia-smi &> /dev/null; then
#         num_gpus=$(nvidia-smi -L | wc -l)
#         gpu_list=$(seq -s, 0 $((num_gpus-1)))
#         else
#         num_gpus=-1
#         gpu_list="-1"
#     fi
#     # You can also manually specify CUDA_VISIBLE_DEVICES
#     # if you don't want to utilize all available GPU resources.
#     export CUDA_VISIBLE_DEVICES="${gpu_list}"
#     echo "Using CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"
# else 
#   echo "CUDA_VISIBLE_DEVICES was set from the outside to: [$CUDA_VISIBLE_DEVICES]"
# fi

# start_stage=0 # start from 0 if you need to start from data preparation
# end_stage=5
model_tag=final

batch_size=1 # default of the original script

decode_modes=
dir=
nj=16
checkpoint=
decode_checkpoint=
# use average_checkpoint will get better result
average_checkpoint=true
# maybe you can try to adjust it if you can not get close results as README.md
average_num=10
decode_modes=ctc_greedy_search
beam_size=10

# Testing
# Specify decoding_chunk_size if it's a unified dynamic chunk trained model
# -1 for full chunk
decoding_chunk_size=
ctc_weight=0.5

train_extra_params=
train_set=train
train_dev=dev
recog_set="test"


train_engine=torch_ddp

deepspeed_config=../../aishell/s0/conf/ds_stage2.json
deepspeed_save_states="model_only"

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0

. tools/parse_options.sh || exit 1;

# Make sure there is one argument.
if [ -z "$config" ]; then
    echo "Config was not set. Exiting."
    usage
    exit 1
fi

if [ "x$init_checkpoint" != "x" ]; then
  if [ "x$checkpoint" == "x" ]; then
     checkpoint="$init_checkpoint"
  fi
fi

export WANDB_PROJECT=$WANDB_PROJECT
echo wandb_project = $WANDB_PROJECT

dir=$base_dir/exp/$exp_name
echo exp_name=$exp_name  dir=$dir

# set_stages $start_stage $end_stage

# set -e
# set -u
# set -o pipefail



# if accept_stage 1; then
# Training
mkdir -p $dir
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
# Use "nccl" if it works, otherwise use "gloo"
dist_backend="nccl"
if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
else
    echo "$0: using torch ddp"
fi

# The total number of processes/gpus, so that the master knows
# how many workers to wait for.
# More details about ddp can be found in
# https://pytorch.org/tutorials/intermediate/dist_tuto.html
world_size=`expr $num_gpus \* $num_nodes`
rdzv_id=$RANDOM
echo "total gpus is: $world_size"
echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
echo "Rendez-Vous ID : $rdzv_id"

#export DNNL_PRIMITIVE_CACHE_CAPACITY=0
torchrun --nnodes=$num_nodes --standalone --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
        --rdzv_id=$rdzv_id --rdzv_backend="c10d" \
wenet/bin/train.py \
  $extra_train_args --config $train_config \
  --data_type $data_type \
  --train_data $training_shards_list \
  --cv_data $dev_shards_list \
  ${checkpoint:+--checkpoint $checkpoint} \
  --model_dir $dir \
  --ddp.dist_backend $dist_backend \
  --num_workers $num_workers \
  --prefetch $prefetch \
  --pin_memory \
  --deepspeed_config ${deepspeed_config} \
  --deepspeed.save_states ${deepspeed_save_states}

record_elapsed "1) Training the model"
# fi

#if accept_stage 2; then
#    timer_start
#
#    #rnd=$RANDOM
#    rnd=$(date +%Y%m%d-%H%M%S)
#    model_tag="avg_${average_num}.${rnd}"
#    decode_checkpoint="$dir/${model_tag}.pt"
#    echo "do model average and final checkpoint is $decode_checkpoint"
#    python wenet/bin/average_model.py \
#        --dst_model $decode_checkpoint \
#        --src_path $dir  \
#        --num ${average_num} \
#        --val_best |& tee "$dir/average_model.${model_tag}.log"
#
#    record_elapsed "2) averaging models"
#fi

# if accept_stage 3; then
# timer_start
#     # Export the best model you want
#     python wenet/bin/export_jit.py \
#         --config $dir/train.yaml \
#         --checkpoint $dir/avg_${average_num}.pt \
#         --output_file $dir/final.zip
#     record_elapsed "3) Export/quantizing the model"
# fi

record_elapsed_since "Total time" ${STAMP0}
dump_timings
