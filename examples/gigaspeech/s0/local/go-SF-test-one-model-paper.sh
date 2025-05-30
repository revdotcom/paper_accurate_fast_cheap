#!/bin/bash

decode_checkpoint=$1
mdir=$(dirname $decode_checkpoint)
bname=$(basename $decode_checkpoint .pt)
config=$mdir/$bname.run-config.yaml
result_dir=$2
gpu=$3

decoding_chunk_size=
ctc_weight=0.3
reverse_weight=0.5
transducer_weight=0.7
# attention_rescoring     

modes="rnnt_beam_search"
for mode in $modes; do
   if [ ! -e $result_dir/$mode/results.wer ]; then
	python wenet/bin/recognize.py --gpu $3 \
	--modes $modes \
	--config $config \
	--data_type "shard" \
	--test_data /shared/speech-shards/test/en/GigaSpeech/gs_test_shards.list  \
	--checkpoint $decode_checkpoint \
	--beam_size 8 \
	--batch_size 16 \
	--length_penalty 0.0 \
	--result_dir $result_dir \
	--reverse_weight $reverse_weight \
	--ctc_weight $ctc_weight \
	--transducer_weight $transducer_weight
   fi 

	python tools/compute-wer.py --char=1 -v=1 /shared/speech-shards/test/en/GigaSpeech/text $result_dir/$mode/text > $result_dir/$mode/results.wer
	tail $result_dir/$mode/results.wer
	python tools/compute-wer-giga.py --char=1 -v=1 /shared/speech-shards/test/en/GigaSpeech/text  $result_dir/$mode/text > $result_dir/$mode/results.filtered.wer

        cat $result_dir/$mode/text | perl -ple 's/<sw>//ig;s/ +/ /g;' > $result_dir/$mode/text_nosw
	python tools/compute-wer-giga.py --char=1 -v=1 /shared/speech-shards/test/en/GigaSpeech/text $result_dir/$mode/text_nosw > $result_dir/$mode/results_nosw.filtered.wer
	tail $result_dir/$mode/results.filtered.wer $result_dir/$mode/results_nosw.filtered.wer
done

