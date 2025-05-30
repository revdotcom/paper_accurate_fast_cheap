# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
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

import copy
import deepspeed
import json
import logging
import os
import math
import torch
import yaml
import wandb
import torch_optimizer as optim_ex

import torch.optim as optim
import torch.distributed as dist

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live)
from deepspeed.runtime.zero.stage3 import (
    estimate_zero3_model_states_mem_needs_all_live)
from deepspeed.utils.zero_to_fp32 import (
    convert_zero_checkpoint_to_fp32_state_dict)
from wenet.dataset.dataset import Dataset
from wenet.utils.checkpoint import save_checkpoint
from wenet.utils.scheduler import WarmupLR, NoamHoldAnnealing, SteadyLR
from wenet.utils.ctc_utils import get_blank_id


def add_model_args(parser):
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument("--enc_init",
                        default=None,
                        type=str,
                        help="Pre-trained model to initialize encoder")
    parser.add_argument(
        '--enc_init_mods',
        default="encoder.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of encoder modules \
                        to initialize ,separated by a comma")
    parser.add_argument(
        '--freeze_modules',
        default="",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help='free module names',
    )
    # JPR, From rev-wenet
    parser.add_argument('--freeze_encoder',
                        action='store_true',
                        default=False,
                        help='Freeze encoder weights')
    parser.add_argument('--freeze_non_lsl',
                        action='store_true',
                        default=False,
                        help='Freeze everything except language-specific-layers')
    parser.add_argument('--restore_encoder_only',
                        action='store_true',
                        default=False,
                        help='Restore only encoders weights from checkpoint')

    # JPR :
    # --no-export is replaced by not passing --jit to the command line
    # parser.add_argument('--no-export',
    #                     action='store_true',
    #                     dest='no_export',
    #                     default=False,
    #                     help='Will skip trying to create a jit version of the model, useful for dev work.')
    parser.add_argument('--load_from_wandb',
                        action='store_true',
                        default=False,
                        help='When set, identifies the last snapshot with an optimizer from'
                             ' the WandB project, downloads it, and uses it as checkpoint.')

    return parser


def add_trace_args(parser):
    parser.add_argument('--jit',
                        action='store_true',
                        default=False,
                        help='if use jit to trace model while training stage')
    parser.add_argument('--print_model',
                        action='store_true',
                        default=False,
                        help='print model')
    return parser


def add_dataset_args(parser):
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    return parser

def add_lora_args(parser):
    parser.add_argument("--use_lora",
                        action="store_true",
                        default=False,
                        help="use the lora finetuning")
    parser.add_argument("--only_optimize_lora",
                        action="store_true",
                        default=False,
                        help="freeze all other parameters and only optimize \
                        LoRA-related prameters.")
    parser.add_argument("--lora_list",
                        default=['o', 'q', 'k', 'v'],
                        help="lora module list.")
    parser.add_argument("--lora_rank",
                        default=8,
                        type=int,
                        help="lora rank num.")
    parser.add_argument("--lora_alpha",
                        default=8,
                        type=int,
                        help="lora scale param, scale=lora_alpha/lora_rank.")
    parser.add_argument("--lora_dropout",
                        default=0,
                        type=float,
                        help="lora dropout param.")
    return parser


def add_ddp_args(parser):
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--fp16_grad_sync',
                        action='store_true',
                        default=False,
                        help='Use fp16 gradient sync for ddp')
    return parser


def add_deepspeed_args(parser):
    parser.add_argument('--timeout',
                        default=30,
                        type=int,
                        help='timeout (in seconds) of wenet_join. ' +
                        '30s for aishell & 300s for wenetspeech')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    # DeepSpeed automaticly add '--deepspeed' and '--deepspeed_config' to parser
    parser = deepspeed.add_config_arguments(parser)
    return parser


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine == "torch_ddp":
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend)
    elif args.train_engine == "deepspeed":
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    else:
        logging.error("not supported engine: {}".format(args.train_engine))
    return world_size, local_rank, rank


def check_modify_and_save_config(args, configs, symbol_table, save=True):
    if args.train_engine == "torch_ddp":
        if args.use_amp:
            configs["dtype"] = "fp16"
        else:
            configs["dtype"] = "fp32"
    elif args.train_engine == "deepspeed":
        # NOTE(xcsong): DeepSpeed does not support uneven data. When using custom
        #   dataset, we need to manually ensure that the data is evenly distributed
        #   across all processe. we impl `train_utils.py::wenet_join` for this func
        #   ref: https://github.com/microsoft/DeepSpeed/issues/2223
        #
        # NOTE(xsong):  We also need to keep:
        #       1. `train_micro_batch_size_per_gpu == 1`
        #       2. `accum_grad (in train_confomrer.yaml)
        #               == gradient_accumulation_steps (in ds_config.json)`
        #       3. `grad_clip (in train_confomrer.yaml)
        #               == gradient_clipping (in ds_config.json)`
        #   The reason for such consistence checking lies in that deepspeed's native
        #   dataloader uses PyTorch's torch.utils.data.DistributedSampler which does
        #   not support IterableDataset, IterableDataset is extremly useful in large
        #   scale training because it lets you stream the data without having to
        #   download the complete dataset.
        #       ref: https://github.com/microsoft/DeepSpeed/issues/1371
        #           https://github.com/microsoft/DeepSpeed/issues/285
        #   To make deepspeed training compatible with IterableDataset, we have to
        #   use custom dataloader instead of deepspeed's native loader and thus we
        #   should configure batchsize in train_confomrer.yaml instead of
        #   ds_config.json. On the contrary, gradient accumulation / clipping should be
        #   configured in ds_config.json since they will be handled by ds automatically.
        #       ref: https://github.com/microsoft/DeepSpeed/issues/62
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs["dtype"] = "bf16"
        else:
            configs["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        assert ds_configs["gradient_accumulation_steps"] == configs[
            'accum_grad']
        assert ds_configs["gradient_clipping"] == configs['grad_clip']
        assert ds_configs["steps_per_print"] == configs['log_interval']

    if args.use_lora:
        configs['encoder_conf']['lora_list'] = args.lora_list
        configs['encoder_conf']['lora_rank'] = args.lora_rank
        configs['encoder_conf']['lora_alpha'] = args.lora_alpha
        configs['encoder_conf']['lora_dropout'] = args.lora_dropout

    if 'input_dim' not in configs:
        if 'fbank_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        elif 'log_mel_spectrogram_conf' in configs['dataset_conf']:
            input_dim = configs['dataset_conf']['log_mel_spectrogram_conf'][
                'num_mel_bins']
        else:
            input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
    else:
        input_dim = configs['input_dim']

    configs, _ = get_blank_id(configs, symbol_table)

    configs['input_dim'] = input_dim
    configs['output_dim'] = configs['vocab_size']

    configs['train_engine'] = args.train_engine
    configs['use_amp'] = args.use_amp
    configs['model_dir'] = args.model_dir
    configs['save_states'] = args.save_states

    # Save configs to model_dir/train.yaml for inference and export
    if save and int(os.environ.get('RANK', 0)) == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    if configs["model_conf"].get("apply_non_blank_embedding", False):
        logging.warn('Had better load a well trained model'
                     'if apply_non_blank_embedding is true !!!')

    return configs


def init_dataset_and_dataloader(args, configs, tokenizer, seed=777, skip_train=False):
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['spec_sub'] = False
    cv_conf['spec_trim'] = False
    cv_conf['shuffle'] = False
    cv_conf['apply_rir'] = False
    cv_conf['apply_telephony'] = False
    cv_conf['fbank_conf']['dither'] = 0

    # let's make sure that the devset won't skip any utterances
    if cv_conf['batch_conf'].get('batch_type', "static") == "distribute":
        cv_conf['batch_conf']['batch_type'] = "dynamic"

    configs['vocab_size'] = tokenizer.vocab_size()
    cv_dataset = Dataset(args.data_type,
                         args.cv_data,
                         tokenizer,
                         cv_conf,
                         partition=False)

    # NOTE(xcsong): Why we prefer persistent_workers=True ?
    #   https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110
    if skip_train:
        train_data_loader = None
        train_dataset = None
    else:
        train_dataset = Dataset(args.data_type, args.train_data, tokenizer,
                                train_conf, True)
        train_data_loader = DataLoader(train_dataset,
                                       batch_size=None,
                                       pin_memory=args.pin_memory,
                                       num_workers=args.num_workers,
                                       persistent_workers=False,
                                       #persistent_workers=True,
                                       generator=generator,
                                       prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                persistent_workers=False,
                                #persistent_workers=True,
                                generator=generator,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def wrap_cuda_model(args, model):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if hasattr(model, 'encoder'):
        grad_ckpt = getattr(model.encoder, 'gradient_checkpointing', False)
    else:
        grad_ckpt = False
    # TODO(xcsong): could one GPU use ddp? and int(os.environ.get('WORLD_SIZE', 1)) > 1
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=not grad_ckpt)
        device = torch.device("cuda")
        if args.fp16_grad_sync:
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default as comm_hooks, )
            model.register_comm_hook(state=None,
                                     hook=comm_hooks.fp16_compress_hook)
    elif args.train_engine == "deepspeed":  # deepspeed
        # NOTE(xcsong): look in detail how the memory estimator API works:
        #   https://deepspeed.readthedocs.io/en/latest/memory.html#discussion
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
            logging.info("Estimating model states memory needs (zero3)...")
            estimate_zero3_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
        device = None  # Init device later
        pass  # Init DeepSpeed later
    else:
        logging.error("not supported engine: {}".format(args.train_engine))

    return model, device


def init_optimizer_and_scheduler(args, configs, model):

    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        if hasattr(model.module, 'teacher'):
            for p in model.module.teacher.parameters():
                p.requires_grad = False
    else:
        if hasattr(model, 'teacher'):
            for p in model.teacher.parameters():
                p.requires_grad = False

    if configs['optim'] == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **configs['optim_conf'])
    elif configs['optim'] == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), **configs['optim_conf'])
    elif configs['optim'] == "novograd":
        optimizer = optim_ex.NovoGrad(filter(lambda p: p.requires_grad, model.parameters()), lr=float(configs['optim_conf']['lr']), betas=(0.8, 0.25))
    else:
        raise ValueError("unknown optimizer: " + configs['optim'])

    # restoring from snapshot if that's the case
    # if 'lr' in configs:
    #     for g in optimizer.param_groups:
    #         g['lr'] = configs.get('lr')

    scheduler_type = None
    if configs['scheduler'] == 'warmuplr':
        scheduler_type = WarmupLR
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    elif configs['scheduler'] == 'NoamHoldAnnealing':
        scheduler_type = NoamHoldAnnealing
        scheduler = NoamHoldAnnealing(optimizer, **configs['scheduler_conf'])
    elif configs['scheduler'] == 'steadylr':
        scheduler = SteadyLR(optimizer, **configs['scheduler_conf'])
    else:
        raise ValueError("unknown scheduler: " + configs['scheduler'])

    step = configs.get("step", -1)
    scheduler.set_step(step)

    # NOTE(xcsong): Custom optimizer might yield poor performance when
    #   zero-offload is enabled, if you do want to offload optimizer to CPU,
    #   please set optimizer in ds_config.json, see:
    #   (https://www.deepspeed.ai/docs/config-json/#optimizer-parameters)
    if args.train_engine == "deepspeed":
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "optimizer" in ds_configs:
            # NOTE(xcsong): Disable custom optimizer if it is set in ds_config,
            # extremely useful when enable cpu_offload, DeepspeedCpuAdam
            # could be 4~5x faster than torch native adam
            optimizer = None
            if "scheduler" in ds_configs:
                scheduler = None
            else:

                def scheduler(opt):
                    return scheduler_type(opt, **configs['scheduler_conf'])

        model, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            model_parameters=model.parameters())
    else:
        if (hasattr(args, 'checkpoint') and args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            print(f"checkpoint keys {checkpoint.keys()}")
            logging.info(f"Checkpoint provided, attempting to restore optimizer and scheduler from {args.checkpoint}")
            if 'optimizer0' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer0'])
                configs['lr'] = optimizer.param_groups[0]['lr']
                logging.info(f"restoring optimizer from checkpoint, optimizer lr {optimizer.param_groups[0]['lr']}")
            else:
                logging.info("optimizer0 not found in checkpoint")

            if 'scheduler0' in checkpoint:
                logging.info("restoring scheduler from checkpoint")
                scheduler.load_state_dict(checkpoint['scheduler0'])
            else:
                logging.info("scheduler0 not found in checkpoint")
        else:
            logging.info("No checkpoint provided, not restoring optimizer and scheduler")

    return model, optimizer, scheduler


def trace_and_print_model(args, model):
    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    if int(os.environ.get('RANK', 0)) == 0:
        if args.jit:
            script_model = torch.jit.script(model)
            script_model.save(os.path.join(args.model_dir, 'init.zip'))
        if args.print_model:
            print(model)
            num_params = sum(p.numel() for p in model.parameters())
            print('the number of model params: {:,d}'.format(num_params))


def init_summarywriter(args):
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        exp_id = os.path.basename(args.model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))
    return writer

# JPR, adapted from rev-wenet train.py
def init_wandb(args, configs):
    # configs is the yaml config
    if int(os.environ.get('RANK', 0)) == 0:
        model_dir = args.model_dir
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        if "WANDB_PROJECT" in os.environ and os.environ["WANDB_PROJECT"] != "":
            wandb_project_id=os.environ["WANDB_PROJECT"]
        else:
            wandb_project_id=exp_id

        WANDB_KEY = os.environ["WANDB_KEY"]
        WANDB_HOST = os.environ["WANDB_HOST"]
        wandb.login(host=WANDB_HOST, key=WANDB_KEY)
        wandb.init(
            project=wandb_project_id,
            config=configs,
            job_type='training'
        )

        artifact = wandb.Artifact('wenet-tree', type='code', description="snapshot of the wenet folders at launch time")
        artifact.add_dir("wenet/")
        wandb.log_artifact(artifact)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

        dataset_artifact = wandb.Artifact('training_data_list', type='train_dataset')
        dataset_artifact.add_file(args.train_data, name='train.list')
        wandb.log_artifact(dataset_artifact)

        dev_data_artifact = wandb.Artifact('dev_data_list', type='dev_dataset')
        dev_data_artifact.add_file(args.cv_data, name='dev.list')
        wandb.log_artifact(dev_data_artifact)

        # TODO (JPR): warning, this assumes a bpe tokenizer, this won't work with any type of tokenizers...
        tokenizer_artifact = wandb.Artifact('tokenizer', type='tokenizer')
        # we need to add the type: char|bpe|whisper|paraformer
        tokenizer_artifact.add_file(configs['tokenizer_conf']['bpe_path'], name="tk.model")
        tokenizer_artifact.add_file(configs['tokenizer_conf']['symbol_table_path'], name="tk.units.txt")
        wandb.log_artifact(tokenizer_artifact)



def save_model(model, info_dict, optimizer=None):
    rank = int(os.environ.get('RANK', 0))
    tag = info_dict["tag"] if 'tag' in info_dict else 'snapshot'
    model_dir = info_dict["model_dir"]
    saving_config = info_dict.get('snapshot_saving_conf', {})
    # save ckpt
    # print(f"save model, rank = {rank}")
    if info_dict["train_engine"] == "deepspeed":
        # NOTE(xcsong): All ranks should call this API, but only rank 0
        #   save the general model params. see:
        #   https://github.com/microsoft/DeepSpeed/issues/2993
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir,
                                  tag=tag,
                                  client_state=info_dict)
            if info_dict["save_states"] == "model_only" and rank == 0:
                convert_zero_checkpoint_to_fp32_state_dict(model_dir,
                                                           "{}/{}.pt".format(
                                                               model_dir, tag),
                                                           tag=tag)
                os.system("rm -rf {}/{}".format(model_dir, tag))
    elif rank == 0:
        # NOTE(xcsong): For torch_ddp, only rank-0 should call this.
        # save_model_path = os.path.join(model_dir, '{}.pt'.format(tag))
        save_checkpoint(model, model_dir, info_dict, snapshot_name=tag, snapshot_conf=saving_config, optimizer=optimizer)
    # save yaml
    if rank == 0:
        with open("{}/{}.yaml".format(model_dir, tag), 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)


def wenet_join(group_join, info_dict):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    train_engine = info_dict.get('train_engine', "torch_ddp")

    if info_dict["batch_idx"] == 0 or train_engine == "torch_ddp":
        # NOTE(xcsong): skip first batch because its processing time includes
        #   dataloader initialization time, which may exceed 30 seconds
        return False

    try:
        # NOTE(xcsong): Why we need a new group?
        #   Because Deepspeed has its own group where all the relevant communication
        #   operations are executed. If we add a communication operation that is not
        #   managed by Deepspeed in this group, it's highly likely to cause
        #   communication chaos, resulting in hard-to-troubleshoot hangs.
        dist.monitored_barrier(group=group_join,
                               timeout=group_join.options._timeout)
    except RuntimeError as e:
        logging.info("Detected uneven workload distribution: {}\n".format(e) +
                     "Break current worker to manually join all workers, " +
                     "world_size {}, current rank {}, current local_rank {}\n".
                     format(world_size, rank, local_rank))
        return True

    return False


def batch_forward(model, batch, scaler, info_dict):
    train_engine = info_dict.get('train_engine', "torch_ddp")
    device = int(os.environ.get('LOCAL_RANK', 0))
    accum_grad = info_dict.get('accum_grad', 1)

    dtype = info_dict.get("dtype", "fp32")
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = None

    if train_engine == "deepspeed":
        # deepspeed
        with torch.cuda.amp.autocast(enabled=dtype is not None,
                                     dtype=dtype,
                                     cache_enabled=False):
            loss_dict = model(batch, device)
    else:
        # torch_ddp
        # autocast context
        # The more details about amp can be found in
        # https://pytorch.org/docs/stable/notes/amp_examples.html
        #print(f">> inf or nan => {torch.isnan(batch['feats']).sum() + torch.isinf(batch['feats']).sum()}  on {batch['feats'].size()}, keys : {batch['keys']}")
        #print(f">> inf or nan => {torch.isnan(batch['feats']).sum() + torch.isinf(batch['feats']).sum()}  on {batch['feats'].size()}")
        with torch.cuda.amp.autocast(scaler is not None):
            loss_dict = model(batch, device)
    info_dict['loss_dict'] = loss_dict

    return info_dict


def batch_backward(model, scaler, info_dict):
    train_engine = info_dict.get("train_engine", "torch_ddp")
    accum_grad = info_dict.get('accum_grad', 1)
    use_amp = info_dict.get('use_amp', False)
    if use_amp:
        assert scaler is not None
    loss = info_dict['loss_dict']['loss']

    if train_engine == "deepspeed":
        # NOTE(xcsong): `model.backward(loss)` is equivalent to
        #               `scale_loss_wrt_accum_grad + loss.backward()`
        #   ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api
        scaled_loss = model.backward(loss)
    elif train_engine == "torch_ddp":
        scaled_loss = loss / accum_grad
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
    info_dict['loss_dict']['loss'] = scaled_loss
    for loss_name, loss_value in info_dict['loss_dict'].items():
        if loss_value is not None:
            if type(loss_value) is float or type(loss_value) is int:
                info_dict['loss_dict'][loss_name] = loss_value
            else:
                info_dict['loss_dict'][loss_name] = loss_value.item()

    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict):
    rank = int(os.environ.get('RANK', 0))
    train_engine = info_dict.get("train_engine", "torch_ddp")
    accum_grad = info_dict.get('accum_grad', 1)
    use_amp = info_dict.get('use_amp', False)
    clip = info_dict.get('grad_clip', 50.0)
    clip_hard_maxvalue = info_dict.get('grad_clip_hard_maxvalue', float('inf'))
    clip_hard_warmup = info_dict.get('grad_clip_hard_warmup', math.inf)
    batch_idx = info_dict["batch_idx"]
    if use_amp:
        assert scaler is not None

    grad_norm = 0.0
    if train_engine == "deepspeed":
        # NOTE(xcsong): The step() function in DeepSpeed engine updates the
        #   model parameters as well as the learning rate.
        #   Zeroing the gradients is handled automatically by
        #   DeepSpeed after the weights have been updated using a mini-batch.
        #   DeepSpeed also performs gradient averaging automatically at the
        #   gradient accumulation boundaries and addresses clip_grad_norm internally.
        #   `ds_model.step() =  clip_grad_norm_() + optimizer.step()
        #                       + optimizer.zero_grad() + scheduler.step()`
        #   ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api
        info_dict["is_gradient_accumulation_boundary"] = \
            model.is_gradient_accumulation_boundary()
        model.step()
        grad_norm = model.get_global_grad_norm()
    elif (batch_idx + 1) % accum_grad == 0:
        # Use mixed precision training
        if use_amp:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), clip)
            # Must invoke scaler.update() if unscale_() is used in
            # the iteration to avoid the following error:
            #   RuntimeError: unscale_() has already been called
            #   on this optimizer since the last update().
            # We don't check grad here since that if the gradient
            # has inf/nan values, scaler.step will skip
            # optimizer.step().
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = clip_grad_norm_(model.parameters(), clip)
            if torch.isfinite(grad_norm):
                if grad_norm <= clip_hard_maxvalue or batch_idx<clip_hard_warmup:
                    optimizer.step()
                else:
                    grad_norm2 = clip_grad_norm_(model.parameters(), clip)
                    logging.warning(f"Grad norm too high: No update for this batch : batch_idx: {batch_idx} grad_norm: {grad_norm.item()} grad_norm2: {grad_norm2.item()}")

        optimizer.zero_grad()
        scheduler.step()
        grad_norm = grad_norm.item()

    info_dict["lr"] = optimizer.param_groups[0]['lr']
    info_dict["grad_norm"] = grad_norm

    return info_dict


def log_per_step(writer, info_dict):
    tag = info_dict["tag"]
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    epoch = info_dict.get('epoch', 0)
    train_engine = info_dict.get("train_engine", "torch_ddp")
    accum_grad = info_dict.get('accum_grad', 1) if tag != "CV" else 1
    log_interval = info_dict.get('log_interval', 10)
    lr = info_dict.get("lr", 0.0)
    is_gradient_accumulation_boundary = info_dict.get(
        "is_gradient_accumulation_boundary", False)

    rank = int(os.environ.get('RANK', 0))

    if tag == "TRAIN" and rank == 0 and writer is not None:
        if (train_engine == "deepspeed" and is_gradient_accumulation_boundary) or \
           (train_engine == "torch_ddp" and (batch_idx + 1) % accum_grad == 0):
            writer.add_scalar('train/train_loss',
                              loss_dict['loss'] * accum_grad, step + 1)
            writer.add_scalar('train/grad_norm', info_dict['grad_norm'],
                              step + 1)
            for name, value in loss_dict.items():
                if name != 'loss' and value is not None:
                    writer.add_scalar('train/{}'.format(name), value, step + 1)
    elif "step_" in tag and rank == 0 and writer is not None:
        writer.add_scalar('global_step/lr', lr, step + 1)
        for name, value in loss_dict.items():
            writer.add_scalar('global_step/{}'.format(name), value, step + 1)

    if tag == "CV" or (batch_idx + 1) % log_interval == 0:
        log_str = '{} Batch {}/{} loss {:.6f} '.format(
            tag, epoch, batch_idx + 1, loss_dict['loss'] * accum_grad)
        for name, value in loss_dict.items():
            if name != 'loss' and value is not None:
                log_str += '{} {:.6f} '.format(name, value)
        if tag == "TRAIN":
            log_str += 'lr {:.8f} grad_norm {:.6f} rank {}'.format(
                lr, info_dict['grad_norm'], rank)

        if rank == 0 and wandb.run is not None:
            nick = "dev/" if tag == "CV" else "train/"
            wandb_log = { nick + 'losses': loss_dict,
                          nick + 'step': step,
                          nick + 'epoch': epoch,
                          nick + 'th_accuracy' : info_dict.get('th_accuracy', 0.0),
                          nick + 'frames_seen_so_far': info_dict.get('num_seen_frames', 0),
                        }

            if nick == "train/":
                wandb_log[nick + "lr"] = lr
                wandb_log[nick + "batch_idx"] = batch_idx
                wandb_log[nick + 'grad_norm'] = info_dict.get('grad_norm', -1.0),

            wandb.log(wandb_log)
        logging.info(log_str)


def log_per_epoch(writer, info_dict):
    epoch = info_dict["epoch"]
    loss_dict = info_dict["loss_dict"]
    if int(os.environ.get('RANK', 0)) == 0:
        writer.add_scalar('epoch/lr', info_dict["lr"], epoch)
        for name, value in loss_dict.items():
            writer.add_scalar('epoch/{}'.format(name), value, epoch)


def freeze_modules(model, args):
    for name, param in model.named_parameters():
        for module_name in args.freeze_modules:
            if module_name in name:
                param.requires_grad = False
                logging.debug("{} module is freezed".format(name))
