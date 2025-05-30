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

import argparse
import io
import logging
import os
import random
import tarfile
import zipfile
import tempfile
import time
import multiprocessing
import shutil
import subprocess
from dataclasses import dataclass
from typing import List

import numpy as np

import torch
import torchaudio
from scipy.io import wavfile

AUDIO_FORMAT_SETS = {'flac', 'mp3', 'ogg', 'opus', 'wav', 'wma'}


@dataclass
class UtteranceInfo:
    segment_key: str
    wav_key: str
    txt: str
    wav_specifier: List[str]
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start

    def __str__(self):
        return "\t".join([self.segment_key,
                          self.wav_key,
                          self.txt,
                          " ".join(self.wav_specifier),
                          str(self.start),
                          str(self.end)])

    @classmethod
    def from_string(cls, s):
        fields = s.split('\t')
        fields[3] = fields[3].split()
        fields[4] = float(fields[4])
        fields[5] = float(fields[5])
        return cls(*fields)


def write_tar_file(data_list: List[UtteranceInfo], tar_file: str, audio_dir: str):
    with tarfile.open(tar_file, "w:gz") as tar:
        for utterance in data_list:
            try:
                if len(utterance.wav_specifier) == 1:
                    wav_path = utterance.wav_specifier[0]
                else:
                    wav_path = os.path.join(audio_dir, f'{utterance.wav_key}.wav')
                # effectively read file using memory mapping
                sample_rate, waveform = wavfile.read(wav_path, mmap=True)
                waveform = np.array(waveform[int(utterance.start * sample_rate):int(utterance.end * sample_rate)])

                # add segment wavefile
                with tempfile.NamedTemporaryFile(dir=audio_dir, suffix='.wav') as fwav:
                    wavfile.write(fwav, sample_rate, waveform)
                    fwav.flush()
                    fwav.seek(0)
                    data = fwav.read()

                    txt_file = utterance.segment_key + '.txt'
                    txt = utterance.txt.encode('utf8')
                    txt_data = io.BytesIO(txt)
                    txt_info = tarfile.TarInfo(txt_file)
                    txt_info.size = len(txt)
                    tar.addfile(txt_info, txt_data)

                    wav_file = utterance.segment_key + '.wav'
                    wav_data = io.BytesIO(data)
                    wav_info = tarfile.TarInfo(wav_file)
                    wav_info.size = len(data)
                    tar.addfile(wav_info, wav_data)
            except Exception as e:
                logging.warning(f"Encountered problem with {utterance.segment_key}, "
                               f"{utterance.wav_key} reading from {utterance.wav_specifier}: {e}")
                continue


def write_zip_file(data_list: List[UtteranceInfo], zip_file: str, audio_dir: str):
    with zipfile.ZipFile(zip_file, mode='w') as zip_f:
        for utterance in data_list:
            try:
                if len(utterance.wav_specifier) == 1:
                    wav_path = utterance.wav_specifier[0]
                else:
                    wav_path = os.path.join(audio_dir, f'{utterance.wav_key}.wav')

                sample_rate, waveform = wavfile.read(wav_path, mmap=True)
                waveform = np.array(waveform[int(utterance.start * sample_rate):int(utterance.end * sample_rate)])

                with tempfile.NamedTemporaryFile(dir=audio_dir, suffix='.wav') as fwav:
                    wavfile.write(fwav, sample_rate, waveform)
                    fwav.flush()
                    fwav.seek(0)
                    data = fwav.read()

                    txt_file = utterance.segment_key + '.txt'
                    txt = utterance.txt.encode('utf8')
                    txt_info = zipfile.ZipInfo(txt_file, time.localtime()[:6])
                    txt_data = io.BytesIO(txt)
                    zip_f.writestr(txt_info, txt)

                    wav_file = utterance.segment_key + '.wav'
                    wav_data = data
                    wav_info = zipfile.ZipInfo(wav_file, time.localtime()[:6])
                    zip_f.writestr(wav_info, wav_data)

            except Exception as e:
                logging.warning(f"Encountered problem with {utterance.segment_key}, "
                               f"{utterance.wav_key} reading from {utterance.wav_specifier}: {e}")
                continue


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--zip-shards', action='store_true', help='write shards in zip format')
    parser.add_argument('input', help='data file with shard info')
    parser.add_argument('shard_file', help='text file')
    parser.add_argument('audio_dir', help='path to written wavs')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    data = []
    with open(args.input) as ifp:
        for line in ifp:
            data.append(UtteranceInfo.from_string(line.strip()))

    if args.zip_shards:
        write_zip_file(data, args.shard_file, args.audio_dir)
    else:
        write_tar_file(data, args.shard_file, args.audio_dir)

if __name__ == '__main__':
    main()
