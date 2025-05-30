#!/usr/bin/env python3

import argparse
import os
import io

import random
from urllib.parse import urlparse
from subprocess import PIPE, Popen
import logging
import boto3

logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('nose').setLevel(logging.CRITICAL)
logging.getLogger('s3transfer').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge and shuffle a series of shards.list files, either sitting on s3 or locally')

    parser.add_argument('--output',
                        required=True,
                        help='output file where to write the merged shards list')
    parser.add_argument('--no-shuffle', action='store_true', help="Turn off shuffling shards list (default is to shuffle)")
    parser.add_argument('inputs', nargs='+',  help='input shards.list files to merge')
    args = parser.parse_args()

    session = boto3.Session(profile_name='rev-inst')
    s3res = session.client('s3', region_name="us-west-2")

    all_shards = []
    for input in args.inputs:
        try:
            pr = urlparse(input)

            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                with open(input, 'r') as f:
                    for line in f:
                        if line.strip() != '':
                            all_shards.append(line.strip())
            elif pr.scheme == 's3':
                buf = io.BytesIO()
                s3res.download_fileobj(pr.netloc, pr.path[1:], buf)
                for line in buf.getvalue().decode('utf-8').split('\n'):
                    all_shards.append(line.strip())
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {input}'
                with Popen(cmd, shell=True, stdout=PIPE) as proc:
                    for line in proc.stdout.read().split('\n'):
                        all_shards.append(line.strip())
        except Exception as ex:
            logging.warning('Failed to open {}'.format(input))
            print(ex)

    if not args.no_shuffle:
        random.shuffle(all_shards)
    logging.info(f"writing {len(all_shards)} to {args.output}")
    with open(args.output, 'w') as f:
        for shard in all_shards:
            if len(shard) == 0:
                continue
            f.write(shard + '\n')


