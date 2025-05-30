# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
# importing the module
#import tracemalloc
import argparse
import datetime
import logging
import os
import torch
import yaml

import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record

from wenet.utils.executor import Executor
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import (
    init_wandb, 
    add_model_args, add_dataset_args, add_ddp_args, add_deepspeed_args,
    add_trace_args, init_distributed, init_dataset_and_dataloader,
    check_modify_and_save_config, init_optimizer_and_scheduler,
    trace_and_print_model, wrap_cuda_model, init_summarywriter, save_model,
    log_per_epoch, add_lora_args)
import wenet.dataset.rev_processor as rev_processor

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser = add_model_args(parser)
    parser = add_dataset_args(parser)
    parser = add_ddp_args(parser)
    parser = add_deepspeed_args(parser)
    parser = add_trace_args(parser)
    parser = add_lora_args(parser)
    args = parser.parse_args()
    if args.train_engine == "deepspeed":
        args.deepspeed = True
        assert args.deepspeed_config is not None
    return args


# NOTE(xcsong): On worker errors, this recod tool will summarize the
#   details of the error (e.g. time, rank, host, pid, traceback, etc).
@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # Set random seed
    torch.manual_seed(777)

    # Read config
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # init tokenizer
    tokenizer = init_tokenizer(configs)

    # Init env for ddp OR deepspeed
    _, _, rank = init_distributed(args)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, tokenizer)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs,
                                           tokenizer.symbol_table)

    init_wandb(args, configs)

    # Init asr model from configs
    model, configs = init_model(args, configs)

    # Check model is jitable & print model archtectures
    trace_and_print_model(args, model)

    # Tensorboard summary
    # JPR:we should consider removing this
    writer = init_summarywriter(args)

    # Dispatch model from cpu to gpu
    model, device = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler = init_optimizer_and_scheduler(
        args, configs, model)
    configs['lr'] = optimizer.param_groups[0]['lr']
    
    # Save checkpoints
    # no need to save the optimizer here
    save_model(model,
               info_dict={
                   **configs,
                   "save_time":
                   datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                   "tag":
                   "init"
               }) 

    # Get executor
    tag = configs["init_infos"].get("tag", "init")
    executor = Executor()
    executor.step = configs["init_infos"].get('step', -1) + int("step_" in tag)
    executor.num_seen_frames = configs.get('num_seen_frames', 0)


    # Init scaler, used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Start training loop
    start_epoch = configs["init_infos"].get('epoch', 0) + int("epoch_" in tag)
    configs.pop("init_infos", None)
    final_epoch = None
    for epoch in range(start_epoch, configs.get('max_epoch', 100)):
        configs['epoch'] = epoch

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(
            epoch, lr, rank))

        dist.barrier(
        )  # NOTE(xcsong): Ensure all ranks start Train at the same time.
        # NOTE(xcsong): Why we need a new group? see `train_utils.py::wenet_join`
        group_join = dist.new_group(
            #backend="nccl", timeout=datetime.timedelta(seconds=args.timeout))
            backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train(model, optimizer, scheduler, train_data_loader,
                       cv_data_loader, writer, configs, scaler, group_join, cmdline_args=args, epoch=epoch)
        logging.debug(f"executor.train() is over on rank {rank}")
        dist.destroy_process_group(group_join)
        logging.debug(f"done destroy_process_group() on rank {rank}")

        dist.barrier(
        logging.debug(f"passed barrier, getting in to executor.cv() on rank {rank}")
        )  # NOTE(xcsong): Ensure all ranks start CV at the same time.
        loss_dict = executor.cv(model, cv_data_loader, configs)
        logging.debug(f"executor.cv() completed on rank {rank}")

        lr = optimizer.param_groups[0]['lr']
        logging.info('End of Epoch {} CV info lr {:.4f} cv_loss {:.4f} rank {} acc {:.4f}'.format(
            epoch, lr, loss_dict["loss"], rank, loss_dict["acc"]))

        frame_to_hour_factor = -1
        feats_type = configs['dataset_conf'].get('feats_type', 'fbank')
        if feats_type == 'fbank' or feats_type == 'mfcc':
            feats_conf = configs['dataset_conf'].get(f'{feats_type}_conf', {})
            offset = feats_conf.get('frame_shift', 10)
            # offset is in ms, we'll ignore the width of the frame, assuming is it 
            # negligeable over the number of offsets
            frame_to_hour_factor = 1000 / offset * 3600.0 # number of frames per hours

        info_dict = {
            'epoch': epoch,
            'lr': lr,
            'step': executor.step,
            'frames_seen_so_far' : executor.num_seen_frames,
            # will be - frames_seen_so_far if we can't parse the right feature type
            'hours_seen_so_far' : executor.num_seen_frames / frame_to_hour_factor,
            'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'tag': f"epoch_{epoch:04d}",
            'loss_dict': loss_dict,

            **configs
        }
        print(f"end of epoch, saving model")
        # save_model(model, info_dict=info_dict, snapshot_conf=None, optimizer=optimizer)
        save_model(model, info_dict=info_dict, optimizer=optimizer)
        log_per_epoch(writer, info_dict=info_dict)

        final_epoch = epoch

    if final_epoch is not None and rank == 0:
        final_model_path = os.path.join(args.model_dir, 'final.pt')
        os.remove(final_model_path) if os.path.exists(
            final_model_path) else None
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # starting the monitoring
    #tracemalloc.start()

    try:
        main()
    finally:
        print("stats")
        print(rev_processor.mystats)
