import argparse
import logging
import os

AUDIO_FORMAT_SETS = {'flac', 'mp3', 'ogg', 'opus', 'wav', 'wma'}


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('wav_file', help='wav.scp file')
    parser.add_argument('audio_dir', help='path for storing output wavs')
    parser.add_argument('wav_cmd_file', help='path for storing wav creation cmds')
    args = parser.parse_args()
    return args


def write_wav_cmds(wav_table, audio_dir, cmd_out):
    done_utts = set()
    # we randomly sort this so all processes won't be waiting for the same file in multiple processes
    with open(cmd_out, 'w') as ofp:
        for wav_key, wav in wav_table.items():
            if len(wav) == 1:
                # path to the file
                suffix = os.path.splitext(wav[0])[1][1:]
                assert suffix in AUDIO_FORMAT_SETS, f'{suffix} not in {AUDIO_FORMAT_SETS}'
            else:
                assert wav[-1] == '|'
                wav = wav[:-1]
                if wav_key not in done_utts:
                    out_dir = os.path.join(audio_dir, os.path.dirname(wav_key))
                    os.makedirs(out_dir, exist_ok=True)
                    wav_out = os.path.join(audio_dir, f'{wav_key}.wav')
                    if not os.path.isfile(wav_out):
                        # run the command and pipe it with sox so header information would appear correct
                        cmd = ' '.join(wav) + f' | sox -t wav -r 16k - {wav_out}'
                        ofp.write(cmd + "\n")
                    done_utts.add(wav_key)
    return


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            if len(arr) != 2:
                logging.warning(f"strange line [{line}]")
                continue
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1].split()

    write_wav_cmds(wav_table, args.audio_dir, args.wav_cmd_file)

if __name__ == '__main__':
    main()