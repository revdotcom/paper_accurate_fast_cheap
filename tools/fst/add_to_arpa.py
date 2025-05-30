#!/usr/bin/env python3
# encoding: utf-8

# Adapted from prepare_dict_and_arpa_cv.py
import argparse
parser = argparse.ArgumentParser(description='Add entries from the lexicon to the arpa if they are not already there.')
parser.add_argument('--arpa-input', required=True, help='input .arpa file')
parser.add_argument('--arpa-output', required=True, help='output .arpa file')
parser.add_argument('--lexicon', required=True, help='input lexicon file')
parser.add_argument('--oov-weight', default=-5.0, help='lm weight to give new entries')

args = parser.parse_args()

unigram_table = dict()
with open(args.arpa_input, 'r', encoding='utf8') as fin:
    for line in fin:
        try:
            weight, unit = line.strip().split()
            unigram_table[unit] = float(weight)
        except:
            continue


lexicon_table = set()
with open(args.lexicon, 'r', encoding='utf8') as fin:
    for line in fin:
        word = line.strip().split()[0]
        lexicon_table.add(word)


with open(args.arpa_output, 'w', encoding='utf8') as fout:
    new_entries = set(lexicon_table) - set(unigram_table.keys())

    fout.write('\\data\\\n')
    fout.write('ngram 1={}\n\n'.format(len(unigram_table) + len(new_entries)))
    fout.write('\\1-grams:\n\n')

    for word, weight in unigram_table.items():
        fout.write('{}\t{}\n'.format(weight, word))

    for word in new_entries:
        fout.write('{}\t{}\n'.format(args.oov_weight, word))

    fout.write('\n\\end\\\n')
