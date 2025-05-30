from __future__ import print_function

import argparse
import copy
import logging
import os
import re
import queue
import threading
from typing import Optional, Dict, List
from pathlib import Path
from collections import defaultdict

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
from wenet.transformer.cmvn import GlobalCMVN
from wenet.utils.cmvn import load_cmvn
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.context_graph import ContextGraph
from wenet.utils.ctc_utils import get_blank_id
from wenet.utils.mask import make_pad_mask, add_optional_chunk_mask
import numpy as np
from pymilvus import MilvusClient
from tools.embeddings.milvus import milvus_worker

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--decoding_chunk_size',
                    type=int,
                    default=-1,
                    help='''decoding chunk size,
                            <0: for decoding, use full chunk.
                            >0: for decoding, use fixed chunk size as set.
                            0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                    type=int,
                    default=-1,
                    help='number of left chunks for decoding')
    parser.add_argument('--cat_embs', type=str, default="")
    parser.add_argument('--output_name', type=str, default=None, help='name of output embeddings file')
    parser.add_argument('--milvus_host', type=str, default=None, help='host of milvus server for storing embeddings')
    parser.add_argument('--model_name', type=str, default=None, help='model name used for storing embeddings')
    parser.add_argument('--milvus_batch_size', type=int, default=None, help='batch size for milvus insert')
    args = parser.parse_args()
    print(args)
    if args.output_name is None and args.milvus_host is None:
        raise ValueError("At least one of output_name and milvus_host must be provided")
    return args

def get_embeddings(
    model,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    decoding_chunk_size: int = -1,
    num_decoding_left_chunks: int = -1,
    cat_embs: Optional[torch.Tensor] = None,
):
    assert speech.shape[0] == speech_lengths.shape[0]
    assert decoding_chunk_size != 0
    encoder = model.encoder
    xs = speech
    xs_lens = speech_lengths
    T = xs.size(1)
    masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
    if encoder.global_cmvn is not None: # replace
        xs = encoder.global_cmvn(xs)
    xs, pos_emb, masks = encoder.embed(xs, masks)
    mask_pad = masks  # (B, 1, T/subsample_rate)
    chunk_masks = add_optional_chunk_mask(xs, masks,
                                          encoder.use_dynamic_chunk,
                                          encoder.use_dynamic_left_chunk,
                                          decoding_chunk_size,
                                          encoder.static_chunk_size,
                                          num_decoding_left_chunks)
    embeddings = []
    for i, layer in enumerate(encoder.encoders):
        xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad, cat_embs=cat_embs)
        rep_vector = torch.mean(xs, 1)
        embeddings.append(rep_vector)
    return embeddings

def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    test_conf = copy.deepcopy(configs['dataset_conf'])

    if not 'filter_conf' in test_conf:
        test_conf['filter_conf'] = {}
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['apply_rir'] = False
    test_conf['apply_telephony'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False

    if 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    elif not 'fbank_conf' in test_conf:
        test_conf['fbank_conf'] = {
            "num_mel_bins": 80,
            "frame_shift": 10,
            "frame_length": 25
        }
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0

    if not 'batch_conf' in test_conf:
        test_conf['batch_conf'] = {}
    test_conf['batch_conf']['batch_size'] = args.batch_size
    test_conf['batch_conf']['batch_type'] = "static"
    if not 'cat_emb_conf' in test_conf:
        test_conf['cat_emb_conf'] = {}
    test_conf['cat_emb_conf']['multi_hot'] = False

    tokenizer = init_tokenizer(configs)
    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           tokenizer,
                           test_conf,
                           partition=False,
                           mode='test')

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    configs['output_dim'] = len(tokenizer.symbol_table)

    # Init asr model from configs
    args.jit = False
    model, configs = init_model(args, configs)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()
    if args.output_name is not None:
        os.mkdir(args.output_name)
    if args.milvus_host is not None:
        milvus_client = MilvusClient(uri=args.milvus_host)
        partition_name = re.sub(r'\W', '_', Path(args.test_data).with_suffix('').name)
        model_name = re.sub(r'\W', '_', args.model_name or args.checkpoint)
        milvus_batch_size = args.milvus_batch_size or 1
        milvus_queues = defaultdict(queue.Queue)
        milvus_workers = {}
        milvus_exception_queue = queue.Queue()
        shutdown_event = threading.Event()
    else:
        milvus_client = None


    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            keys = batch["keys"]
            shard_names = ["test" * len(keys)] # batch["shard_file_names"]
            feats = batch["feats"].to(device)
            target = batch["target"].to(device)

            # script argument overrides categories in shard
            if len(args.cat_embs) > 0:
                cat_embs = torch.tensor(
                    [float(c) for c in args.cat_embs.split(',')]).to(device)
            elif "cat_emb" in batch:
                cat_embs = batch["cat_emb"]
            else:
                cat_embs = torch.tensor([1] + [0] *
                                        (len(model.cat_labels) - 1)).to(device)

            feats_lengths = batch["feats_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            infos = {"tasks": batch["tasks"], "langs": batch["langs"]}
            try:
                embeddings = get_embeddings(
                    model,
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    cat_embs=cat_embs)
                # emb is a batch of embeddings from encoder layer i
                for i, emb in enumerate(embeddings):
                    emb = emb.cpu()
                    batched_embeddings = emb.numpy().reshape(len(keys), -1)
                    if args.output_name is not None:
                        with open(f"{args.output_name}/embeddings_layer_{i}", 'a') as f:
                            for emb in batched_embeddings:
                                emb = emb.reshape(1, -1)
                                np.savetxt(f, emb, delimiter=',')
                    if milvus_client is not None:
                        if milvus_exception_queue.qsize() > 0:
                            raise milvus_exception_queue.get()
                        collection_name = f"{model_name}_embeddings_layer_{i}"
                        milvus_queues[(collection_name, partition_name)].put((batched_embeddings, keys))
                        if (collection_name, partition_name) not in milvus_workers:
                            worker = threading.Thread(
                                target=milvus_worker,
                                args=(
                                    milvus_client,
                                    collection_name,
                                    partition_name,
                                    milvus_queues[(collection_name, partition_name)],
                                    milvus_exception_queue,
                                    milvus_batch_size,
                                    shutdown_event,
                                ),
                            )
                            worker.start()
                            milvus_workers[(collection_name, partition_name)] = worker

                if args.output_name is not None:
                    with open(f"{args.output_name}/sample_names", 'a') as f:
                        for sample_name, shard_name in zip(keys, shard_names):
                            f.write(f"{sample_name},{shard_name}")
                            f.write('\n')


                if batch_idx >= 10_000:
                    csv_file_counter += 1
            except Exception as e:
                print(f"Error: {e}")
                break
        if milvus_client is not None:
            shutdown_event.set()
            for q in milvus_queues.values():
                q.join()
            for worker in milvus_workers.values():
                worker.join()


if __name__ == '__main__':
    main()
