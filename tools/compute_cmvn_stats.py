#!/usr/bin/env python3
# encoding: utf-8

import sys
import argparse
import json
import codecs
import yaml

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader
import wenet.dataset.audio_utils as audio_utils

torchaudio.set_audio_backend("sox_io")
import tqdm

def write_cmvn_info(all_mean_stat, all_var_stat, all_number, out_cmvn):
    cmvn_info = {
        'mean_stat': list(all_mean_stat.tolist()),
        'var_stat': list(all_var_stat.tolist()),
        'frame_num': all_number
    }

    with open(out_cmvn, 'w') as fout:
        fout.write(json.dumps(cmvn_info))

class CollateFunc(object):
    ''' Collate function for AudioDataset
    '''

    def __init__(self, feat_dim, resample_rate):
        self.feat_dim = feat_dim
        self.resample_rate = resample_rate
        self.missing_wavs = 0
        self.debug = False
        pass

    def __call__(self, batch):
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        for item in batch:
            value = item[1].strip().split(",")
            assert len(value) == 3 or len(value) == 1
            wav_path = value[0]
            # print(f"wav_path [{wav_path}]")
            # sample_rate = torchaudio.backend.sox_io_backend.info(wav_path).sample_rate
            waveform, sample_rate = audio_utils.get_wavdata_and_samplerate(wav_path)
            if waveform is None :
                self.missing_wavs += 1
                if self.debug:
                    print(f"wav path problematic (#{self.missing_wavs}): {wav_path}")
                continue
            resample_rate = sample_rate
            # len(value) == 3 means segmented wav.scp,
            # len(value) == 1 means original wav.scp
            if len(value) == 3:
                start_frame = int(float(value[1]) * sample_rate)
                end_frame = int(float(value[2]) * sample_rate)
                waveform, sample_rate = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_path,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                pass
                # waveform, sample_rate = torchaudio.load(item[1])

            waveform = waveform * (1 << 15)
            if self.resample_rate != 0 and self.resample_rate != sample_rate:
                resample_rate = self.resample_rate
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate)(waveform)

            # print(f"waveform shape={waveform.shape}", flush=True)
            window_min_size = 4000
            window_size = 15000000 # 15e6 samples
            pos = 0
            while pos < waveform.shape[1]:
                end = pos + window_size
                if waveform.shape[1] - pos < window_min_size:
                    # print(f"{waveform.shape[1] - end} < {window_min_size}, skipping this block")
                    break
                x = waveform[:,pos:end]
                pos = end
                # print(f"         shape={x.shape}", flush=True)
                mat = kaldi.fbank(x,
                                num_mel_bins=self.feat_dim,
                                dither=0.0,
                                energy_floor=0.0,
                                sample_frequency=resample_rate)
                mean_stat += torch.sum(mat, axis=0)
                var_stat += torch.sum(torch.square(mat), axis=0)
                number += mat.shape[0]
        return number, mean_stat, var_stat


class AudioDataset(Dataset):

    def __init__(self, data_file):
        self.items = []
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split(maxsplit=1)
                self.items.append((arr[0], arr[1]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract CMVN stats')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for processing')
    parser.add_argument('--train_config',
                        default='',
                        help='training yaml conf')
    parser.add_argument('--in_scp', default=None, help='wav scp file')
    parser.add_argument('--out_cmvn',
                        default='global_cmvn',
                        help='global cmvn file')

    doc = "Print log after every log_interval audios are processed."
    parser.add_argument("--log_interval", type=int, default=1000, help=doc)
    args = parser.parse_args()

    with open(args.train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    feat_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    resample_rate = 0
    if 'resample_conf' in configs['dataset_conf']:
        resample_rate = configs['dataset_conf']['resample_conf']['resample_rate']
        print('using resample and new sample rate : {}'.format(resample_rate))

    collate_func = CollateFunc(feat_dim, resample_rate)
    dataset = AudioDataset(args.in_scp)
    batch_size = 20
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             sampler=None,
                             num_workers=args.num_workers,
                             collate_fn=collate_func)

    with torch.no_grad():
        all_number = 0
        all_mean_stat = torch.zeros(feat_dim)
        all_var_stat = torch.zeros(feat_dim)
        wav_number = 0
        sz = len(dataset)
        pbar = tqdm.tqdm(total=sz, desc="rows processed")
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat = batch
            all_mean_stat += mean_stat
            all_var_stat += var_stat
            all_number += number
            wav_number += batch_size

            if wav_number % args.log_interval == 0:
                print(f'processed {wav_number} wavs, {all_number} frames',
                      file=sys.stderr,
                      flush=True)


    write_cmvn_info(all_mean_stat, all_var_stat, all_number, args.out_cmvn)