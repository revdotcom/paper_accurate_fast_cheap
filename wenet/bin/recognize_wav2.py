# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
# Copyright (c) 2024 Rev.com (authors: Nishchal Bhandari)
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

import argparse
import copy
import logging
import os
from itertools import groupby, chain
from pathlib import Path
from math import ceil
from typing import Generator, List
import yaml
from time import process_time
import time

import torch
import torch.nn.functional as F

import torchaudio
from torchaudio.compliance import kaldi

from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.search import DecodeResult
from wenet.text.rev_bpe_tokenizer import RevBpeTokenizer
from wenet.utils.cmvn import load_cmvn
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.ctc_utils import get_blank_id
from wenet.bin.ctc_align import ctc_align, adjust_model_time_offset
import traceback

class GpuMemoryMonitor:
    """Monitor GPU memory usage and write to file periodically."""
    
    def __init__(self, filename: str, gpu_id: int = 0, interval_secs: int = 10):
        """Initialize the GPU memory monitor.
        
        Args:
            filename: Path to output file to write memory stats
            gpu_id: GPU ID to monitor (-1 for CPU)
        """
        self.filename = filename
        self.gpu_id = gpu_id
        self.running = False
        self.monitor_thread = None
        self.interval_secs = interval_secs
    def _monitor_gpu(self):
        if self.gpu_id < 0:  # CPU mode
            return
            
        try:
            with open(self.filename, 'w') as f:
                while self.running:
                    # Get memory in MB
                    allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
                    reserved = torch.cuda.memory_reserved(self.gpu_id) / 1024 / 1024
                    
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f'{timestamp} Allocated: {allocated:.1f}MB Reserved: {reserved:.1f}MB\n')
                    f.flush()
                    time.sleep(self.interval_secs)
        except Exception as e:
            print(f"GPU monitoring failed: {str(e)}")
            return

    def start(self):
        """Start monitoring GPU memory usage in background thread."""
        import threading
        import time
        
        if self.monitor_thread is not None:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_gpu, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop the GPU memory monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None


def get_args():
    parser = argparse.ArgumentParser(
        description="Run automatic speech recognition on a given wav file using the Rev model."
    )
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--audio_file", required=True, help="Audio to transcribe")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="gpu id for this rank, -1 for cpu"
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument("--tokenizer-symbols", help="Path to tk.units.txt. Overrides the config path.")
    parser.add_argument("--bpe-path", help="Path to tk.model. Overrides the config path.")
    parser.add_argument("--cmvn-path", help="Path to cmvn. Overrides the config path.")
    parser.add_argument(
        "--rwkv_r", type=int, help="receptance field")
    parser.add_argument(
        "--beam_size", type=int, default=10, help="beam size for search"
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.0,
        help="length penalty for attention decoding and joint decoding modes",
    )
    parser.add_argument(
        "--blank_penalty", type=float, default=0.0, help="blank penalty"
    )
    parser.add_argument("--result_dir", required=True, help="asr result file")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of chunks that are decoded in parallel",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2051,
        help="Size of each chunk that is decoded, in frames",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "attention",
            "ctc_greedy_search",
            "ctc_prefix_beam_search",
            "attention_rescoring",
            "joint_decoding",
            "rnnt_greedy_search",
            "rnnt_beam_search",
            "rnnt_beam_attn_rescoring",
            "ctc_beam_td_attn_rescoring",
            "hlg_onebest",
            "hlg_rescore",
            "paraformer_greedy_search",
            "paraformer_beam_search",
        ],
        default=["attention_rescoring"],
        help="One or more supported decoding mode.",
    )
    parser.add_argument(
        "--ctc_weight",
        type=float,
        default=0.1,
        help="ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode \
                              ctc weight for joint decoding mode",
    )

    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument(
        "--decoding_chunk_size",
        type=int,
        default=-1,
        help="""decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here""",
    )
    parser.add_argument(
        "--num_decoding_left_chunks",
        type=int,
        default=-1,
        help="number of left chunks for decoding",
    )
    parser.add_argument(
        "--simulate_streaming", action="store_true", help="simulate streaming inference"
    )
    parser.add_argument(
        "--reverse_weight",
        type=float,
        default=0.0,
        help="""right to left weight for attention rescoring
                                decode mode""",
    )
    parser.add_argument("--encoder_context_size",
                        type=int,
                        default=-1,
                        help='Limited context size for attention. -1 means unlimited.')
    parser.add_argument("--encoder_global_tokens",
                        type=int,
                        default=0,
                        help='Number of global tokens to use for attention. 0 means no global tokens.')
    parser.add_argument("--encoder_global_tokens_spacing",
                        type=int,
                        default=1,
                        help='Spacing between global tokens. 1 means no spacing.')

    parser.add_argument(
        "--overwrite_cmvn",
        action="store_true",
        help="overwrite CMVN params in model with those in config file",
    )

    parser.add_argument(
        "--verbatimicity",
        type=float,
        default=1.0,
        help="The level of verbatimicity to run the mode. 0.0 would be nonverbatim, and 1.0 would be verbatim. This value gets passed to the LSL layers.",
    )

    parser.add_argument(
        "--timings_adjustment",
        type=float,
        default=230,
        help="Subtract timings_adjustment milliseconds from each timestamp")

    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Choose logging level for statistics and debugging.",
    )
    parser.add_argument(
        "--half", action="store_true", help="will run call model.half()"
    )
    parser.add_argument(
        "--bf16", action="store_true", help="will run call model.to(dtype=torch.bfloat16)"
    )
    parser.add_argument(
        "--tf32", action="store_true", help="enable all TF32 kernels"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(filename)s %(levelname)s: %(message)s",
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    audio_file = args.audio_file
    test_conf = copy.deepcopy(configs.get("dataset_conf", {}))

    if not "filter_conf" in test_conf:
        test_conf["filter_conf"] = {}

    if "mfcc_conf" in test_conf:
        test_conf["mfcc_conf"]["dither"] = 0.0
    elif not "fbank_conf" in test_conf:
        test_conf["fbank_conf"] = {
            "num_mel_bins": 80,
            "frame_shift": 10,
            "frame_length": 25,
        }
    if "fbank_conf" in test_conf:
        test_conf["fbank_conf"]["dither"] = 0.0

    input_frame_length = test_conf["fbank_conf"]["frame_shift"]
    frame_downsampling_factor = {"linear": 1, "conv2d": 4, "conv2d6": 6, "conv2d8": 8}
    output_frame_length = input_frame_length * frame_downsampling_factor.get(
        configs["encoder_conf"]["input_layer"], 4
    )

    if args.cmvn_path:
        configs["cmvn_conf"]["cmvn_file"] = args.cmvn_path
    else:
        # Check if symbol table path is relative or absolute
        cmvn_path = Path(configs["cmvn_conf"]["cmvn_file"])
        if not cmvn_path.is_absolute():
            # Assume it's adjacent to the model
            configs["cmvn_conf"]["cmvn_file"] = (Path(args.checkpoint).parent / cmvn_path).as_posix()

    if args.tokenizer_symbols:
        configs["tokenizer_conf"]["symbol_table_path"] = args.tokenizer_symbols
    else:
        # Check if symbol table path is relative or absolute
        sym_path = Path(configs["tokenizer_conf"]["symbol_table_path"])
        if not sym_path.is_absolute():
            # Assume it's adjacent to the model
            configs["tokenizer_conf"]["symbol_table_path"] = (Path(args.checkpoint).parent / sym_path).as_posix()

    if args.bpe_path:
        configs["tokenizer_conf"]["bpe_path"] = args.bpe_path
    else:
        # Check if bpe model path is relative or absolute
        bpe_path = Path(configs["tokenizer_conf"]["bpe_path"])
        if not bpe_path.is_absolute():
            # Assume it's adjacent to the model
            configs["tokenizer_conf"]["bpe_path"] = (Path(args.checkpoint).parent / bpe_path).as_posix()
    tokenizer = init_tokenizer(configs)

    feats = compute_feats(
        audio_file,
        num_mel_bins=test_conf["fbank_conf"]["num_mel_bins"],
        frame_length=test_conf["fbank_conf"]["frame_length"],
        frame_shift=test_conf["fbank_conf"]["frame_shift"],
        dither=test_conf["fbank_conf"]["dither"],
    )  # Shape is (1, num_frames, num_mel_bins)

    # Pad and reshape into chunks and batches
    def feats_batcher(infeats, chunk_size, batch_size, device):
        batch_num_feats = chunk_size * batch_size
        num_batches = ceil(infeats.shape[1] / batch_num_feats)
        for b in range(num_batches):
            feats_batch = infeats[
                :, b * batch_num_feats : b * batch_num_feats + batch_num_feats, :
            ]
            feats_lengths = torch.tensor([chunk_size] * batch_size, dtype=torch.int32)
            if b == num_batches - 1:
                # last batch can be smaller than batch size
                last_batch_size = ceil(feats_batch.shape[1] / chunk_size)
                last_batch_num_feats = chunk_size * last_batch_size
                feats_lengths = torch.tensor([chunk_size] * last_batch_size, dtype=torch.int32)
                # Apply padding if needed
                pad_amt = last_batch_num_feats - feats_batch.shape[1]
                if pad_amt > 0:
                    feats_lengths = torch.tensor(
                        [chunk_size] * last_batch_size, dtype=torch.int32
                    )
                    feats_lengths[-1] -= pad_amt
                    feats_batch = F.pad(
                        input=feats_batch,
                        pad=(0, 0, 0, pad_amt, 0, 0),
                        mode="constant",
                        value=0,
                    )
            yield feats_batch.reshape(
                -1, chunk_size, test_conf["fbank_conf"]["num_mel_bins"]
            ), feats_lengths.to(device)

    configs["output_dim"] = len(tokenizer.symbol_table)

    if args.encoder_context_size > 0:
        configs["encoder_conf"]["selfattention_layer_type"] = "limited_rel_selfattn"
        configs["encoder_conf"]["att_context_size"] = [args.encoder_context_size, args.encoder_context_size]
        if args.encoder_global_tokens > 0:
            configs["encoder_conf"]["global_tokens"] = args.encoder_global_tokens
            configs["encoder_conf"]["global_tokens_spacing"] = args.encoder_global_tokens_spacing
        else:
            configs["encoder_conf"]["global_tokens"] = 0
            configs["encoder_conf"]["global_tokens_spacing"] = 1

    if args.rwkv_r :
        configs["encoder_conf"]["rwkv_ctx_len"] = args.rwkv_r


    # Init asr model from configs
    args.jit = False
    model, configs = init_model(args, configs)

    # from merge section (JPR)
    if args.overwrite_cmvn and (configs["cmvn_conf"]["cmvn_file"] is not None):
        mean, istd = load_cmvn(configs["cmvn_conf"]["cmvn_file"], configs["cmvn_conf"]["is_json_cmvn"])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(), torch.from_numpy(istd).float()
        )
        model.encoder.global_cmvn = global_cmvn

    print(model.encoder.global_cmvn.mean.shape)

    #model = torch.compile(model, mode="max-autotune", dynamic=True)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    # device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    device = torch.device(f"cuda" if use_cuda else "cpu")
    torch.set_default_device(device)

    _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    logging.info("blank_id is {}".format(blank_id))

    if args.half:
        print("moving the model to half precision")
        model = model.half()
        #feats = feats.half()
    elif args.bf16:
        print("moving the model to bf16 precision")
        model = model.to(dtype=torch.bfloat16)
        #feats = feats.to(dtype=torch.bfloat16)
        print(model)
    elif args.tf32:
        print("turning on all tf32 opts")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True # This is true by default.

    model = model.to(device)
    model.eval()
    feats = feats.to(device)

    files = {}
    timing_files = {}
    for mode in args.modes:
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        timing_filename = Path(dir_name) / (Path(audio_file).with_suffix(".rtf").name)
        memory_filename = Path(dir_name) / (Path(audio_file).with_suffix(".vram").name)
        timing_files[mode] = timing_filename
        monitor = GpuMemoryMonitor(memory_filename, 0)
        monitor.start()
        file_name = os.path.join(dir_name, "text")
        file_name = Path(dir_name) / (Path(audio_file).with_suffix(".ctm").name)
        files[mode] = file_name
    max_format_len = max([len(mode) for mode in args.modes])
    with torch.no_grad():
        # script argument overrides categories in shard
        cat_embs = torch.tensor([args.verbatimicity, 1.0 - args.verbatimicity]).to(
            device
        )

        results = []
        timings = []
        num_frames = []
        i = 0
        for feats_batch, feats_lengths in feats_batcher(
            feats, args.chunk_size, args.batch_size, device
        ):
            print(f"batch {i}")
            i += 1

            # t0 = process_time()
            t0 = time.time()
            hyps = model.decode(
                    args.modes,
                    feats_batch,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    transducer_weight=args.transducer_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight,
                    context_graph=None,
                    blank_id=blank_id,
                    blank_penalty=args.blank_penalty,
                    length_penalty=args.length_penalty,
                    infos={"tasks": ["transcribe"], "langs": ["en"]},
                    cat_embs=cat_embs,
                    )
            print(f"hyps keys = {hyps.keys()}")
            t1 = time.time()
            # t1 = process_time()
            results.append(hyps)
            timings.append(t1 - t0)
            num_frames.append(feats_lengths.sum())
        monitor.stop()


    #print(hyps)
    for mode in args.modes:
        with files[mode].open(mode="w") as fp:
            mode_hyps = chain(*(hyp[mode] for hyp in results))
            fp.write(
                "\n".join(
                    hyps_to_ctm(
                        tokenizer, Path(audio_file).name, list(mode_hyps), args.timings_adjustment, args.chunk_size, input_frame_length, output_frame_length
                    )
                )
            )
        with timing_files[mode].open(mode="w") as fp:
            total_elapsed = 0.0
            total_frames = 0
            for elapsed, num_frames in zip(timings, num_frames):
               local_rtf = elapsed / (num_frames / 100.0)
               total_elapsed += elapsed
               total_frames += num_frames
               fp.write(f"{num_frames=:8d}, {elapsed=:.4f}, {local_rtf=:.4f}\n")
            fp.write(f"{total_frames=:8d}, {total_elapsed=:.4f}, final_rtf {total_elapsed / (total_frames / 100.00):.4f}\n")



def compute_feats(
    audio_file: str,
    resample_rate: int = 16000,
    device="cpu",
    num_mel_bins=23,
    frame_length=25,
    frame_shift=10,
    dither=0.0,
) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
    logging.info(f"detected sample rate: {sample_rate}")
    waveform = waveform.to(torch.float)
    if sample_rate != resample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate
        )(waveform)
    waveform = waveform.to(device)
    feats = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        energy_floor=0.0,
        sample_frequency=resample_rate,
    )
    feats = feats.unsqueeze(0)
    return feats


def hyps_to_ctm(
    tokenizer: RevBpeTokenizer,
    audio_name: str,
    hyps: List[DecodeResult],
    timings_adjustment_ms: int,
    chunk_size: int,
    input_frame_length: int,
    output_frame_length: int
) -> Generator[str, None, None]:
    """Convert a given set of decode results for a single audio file into CTM lines."""

    times = []
    confidences = []
    time_shift_ms = 0
    for hyp in hyps:
        try:
            path = ctc_align(hyp.tokens, hyp.times, hyp.tokens_confidence,  tokenizer,
                            output_frame_length, time_shift_ms)
            path = adjust_model_time_offset(path, timings_adjustment_ms)
            time_shift_ms += chunk_size * input_frame_length
            ctm_lines = []
            for line in path:
                start_seconds = line['start_time_ms'] / 1000
                duration_seconds = line['end_time_ms'] / 1000 - start_seconds
                ctm_line = f"{audio_name} 0 {start_seconds:.2f} {duration_seconds:.2f} {line['word']} {line['confidence']:.2f}"
                yield ctm_line
        except Exception as ex:
            print("exception caught")
            traceback.print_exc()
            pass


if __name__ == "__main__":
    main()
