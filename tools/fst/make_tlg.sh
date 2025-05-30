#!/bin/bash
#

if [ -f path.sh ]; then . path.sh; fi

arpa_lm=$1
lang_dir=$2

[ ! -e $arpa_lm ] && echo No such file $arpa_lm && exit 1;

#   grep -v -i '<unk>' | \
#   grep -v -i '<spoken_noise>' | \
# Compose the language model to FST
cat $arpa_lm | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   arpa2fst --read-symbol-table=$lang_dir/words.txt --keep-symbols=true - | fstprint | \
   tools/fst/eps2disambig.pl | tools/fst/s2eps.pl | fstcompile --isymbols=$lang_dir/words.txt \
     --osymbols=$lang_dir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > $lang_dir/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic $lang_dir/G.fst

# Compose the token, lexicon and language-model FST into the final decoding graph
fsttablecompose $lang_dir/L.fst $lang_dir/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstarcsort --sort_type=ilabel > $lang_dir/LG.fst || exit 1;
fsttablecompose $lang_dir/T.fst $lang_dir/LG.fst > $lang_dir/TLG.fst || exit 1;

echo "Composing decoding graph TLG.fst succeeded"
#rm -r $lang_dir/LG.fst   # We don't need to keep this intermediate FST
