from pathlib import Path
from argparse import ArgumentParser


def _init_args():
    parser = ArgumentParser(description="Removes lines from a file that don't occur in another file based on the first value")
    parser.add_argument('base_file', type=Path,
                        help="File with correct lines we want to keep")
    parser.add_argument('extra_file', type=Path,
                        help="File with extra lines we don't want to keep")
    return parser.parse_args()


if __name__ == '__main__':
    args = _init_args()

    utterances_of_interest = set()
    with args.base_file.open('r') as bfile:
        for line in bfile:
            utt = line.split()[0]

            utterances_of_interest.add(utt)


    with args.extra_file.open('r') as efile:
        for line in efile:
            utt = line.split()[0]

            if utt in utterances_of_interest:
                print(line.strip())
