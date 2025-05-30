#!/usr/bin/env python3
# encoding: utf-8

import argparse
parser = argparse.ArgumentParser(description='create lexicon and .arpa file for CV FST')
parser.add_argument('--tokenizer_units', default="~jp/NeMo/rev-wenet/examples/rev/s0/exp/tokenizers/tokenizer_unigram_10000_8/tk.units.txt", help='tokenizer units file')
parser.add_argument('--tokenizer_model', default="~jp/NeMo/rev-wenet/examples/rev/s0/exp/tokenizers/tokenizer_unigram_10000_8/tk.model", help='tokenizer model')
parser.add_argument('--lm', help='lm .arpa file')
parser.add_argument('--arpa_output', required=True, help='output .arpa file')
parser.add_argument('--lexicon', default="/shared/experiments/NERD-1643/lms/words.no_eps.txt", help='input lexicon file')
parser.add_argument('--lexicon_output', required=True, help='output lexicon file')
parser.add_argument('--cv', help='text file containing cv phrases per line')
parser.add_argument('--alternate_spellings', help='text file containing alternate spellings')
parser.add_argument('--non_cv_weight', default=-5.0, type=float, help='non-cv weight')
parser.add_argument('--non_cv_discount', default=0.0, type=float, help='non-cv discount')
parser.add_argument('--cv_weight', default=-1.0, type=float, help='cv word weight')
parser.add_argument('--phrase_cv_weight', type=float, help='cv phrase weight (defaults to cv_weight)')
parser.add_argument('--n_cv_tokenizations', default=0, type=int, help='number of subword tokenizations of CV terms. n <= 0 means vitterbi best')
parser.add_argument('--cv_phrases', dest='cv_phrases', action='store_true')
parser.add_argument('--no_cv_phrases', dest='cv_phrases', action='store_false')
parser.add_argument('--split_cv_phrases', dest='split_cv_phrases', action='store_true')
parser.add_argument('--no_split_cv_phrases', dest='split_cv_phrases', action='store_false')
parser.set_defaults(cv_phrases=True)
parser.set_defaults(split_cv_phrases=True)

args = parser.parse_args()

unit_table = set()
with open(args.tokenizer_units, 'r', encoding='utf8') as fin:
    for line in fin:
        unit = line.split()[0]
        unit_table.add(unit)

unigram_table = dict()
if args.lm is not None:
    with open(args.lm, 'r', encoding='utf8') as fin:
        for line in fin:
            try:
                weight, unit = line.strip().split()
                unigram_table[unit] = float(weight)
            except:
                continue

def contain_oov(units):
    for unit in units:
        if unit not in unit_table:
            return True
    return False

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load(args.tokenizer_model)
lexicon_table = set()
cv_table = set()
cv_phrase_table = set()
if args.phrase_cv_weight is None:
    args.phrase_cv_weight = args.cv_weight
    
def process_word(word, ignore_lexicon=False, ntok=0):
    if word == '<SPOKEN_NOISE>':
        return None
    elif word == '#0':
        return None
    else:
        # each word only has one pronunciation for e2e system
        if not ignore_lexicon and word in lexicon_table:
            return None
        
        if ntok <= 0:
            toks = [sp.EncodeAsPieces(word)]
        else:
            toks = sp.NBestEncodeAsPieces(word, ntok)
            
        if contain_oov(toks[0]):
            print(
                'Ignoring words {}, which contains oov unit'.format(
                    ''.join(word).strip('▁'))
            )
            return None
        return [' '.join(pieces) for pieces in toks]
            
    
with open(args.lexicon_output, 'w', encoding='utf8') as fout:
    with open(args.cv, 'r', encoding='utf8') as fin:
        for line in fin:
            if line.strip() == "":
                continue
            
            if len(line.split()) > 1 and args.cv_phrases:            
                word = '▁'.join(line.split()).lower()
                cv_phrase_table.add(word)
                toks = process_word(word, ntok=args.n_cv_tokenizations)
                if toks is not None:
                    for chars in toks:
                        fout.write('{} {}\n'.format(word, chars))
                    lexicon_table.add(word)

            if not args.split_cv_phrases and len(line.split()) > 1:
                continue
            
            for word in line.split():
                word = word.lower()
                cv_table.add(word)
                toks = process_word(word, ntok=args.n_cv_tokenizations)
                if toks is not None:
                    for chars in toks:
                        fout.write('{} {}\n'.format(word, chars))
                    lexicon_table.add(word)
    if args.alternate_spellings is not None:
        with open(args.alternate_spellings, 'r', encoding='utf8') as fin:
            for line in fin:
                cv_term, alternate = line.lower().strip().split(" <-> ")
                cv_word = '▁'.join(cv_term.split())
                toks = process_word(alternate, True, ntok=args.n_cv_tokenizations)
                if toks is not None:
                    for chars in toks:
                        fout.write('{} {}\n'.format(cv_word, chars))
    with open(args.lexicon, 'r', encoding='utf8') as fin:
        for line in fin:
            word = line.split()[0]
            chars = process_word(word)
            if chars is not None:
                fout.write('{} {}\n'.format(word, chars[0]))
                lexicon_table.add(word)

print(len(lexicon_table))
print(len(cv_table))
print(len(cv_phrase_table))

with open(args.arpa_output, 'w', encoding='utf8') as fout:
    fout.write('\\data\\\n')
    fout.write('ngram 1={}\n\n'.format(len(lexicon_table) + 4))
    fout.write('\\1-grams:\n\n')

    for word in lexicon_table:
        if word in cv_phrase_table:
            fout.write('{}\t{}\n'.format(args.phrase_cv_weight, word))
        elif word in cv_table:
            fout.write('{}\t{}\n'.format(args.cv_weight, word))
        elif word in unigram_table:
            fout.write('{}\t{}\n'.format(unigram_table[word] + args.non_cv_discount, word))
        else:
            fout.write('{}\t{}\n'.format(args.non_cv_weight + args.non_cv_discount, word))

    fout.write('\n\\end\\\n')
