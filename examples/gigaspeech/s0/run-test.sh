#!/bin/bash

source path.sh

decode_checkpoint=$1
mdir=$(dirname $decode_checkpoint)
config_yaml=$2
result_dir=$3
gpu=$4

decoding_chunk_size=
ctc_weight=0.3
transducer_weight=0.7
reverse_weight=0.5
# attention_rescoring     
# modes="ctc_greedy_decoding attention_rescoring"
# modes="ctc_greedy_decoding"
# modes="attention_rescoring"
modes="rnnt_beam_search"
for mode in $modes; do
	python wenet/bin/recognize.py --gpu $gpu \
	--modes $modes \
	--config $config_yaml \
	--data_type "shard" \
	--test_data /shared/speech-shards/dev/en/GigaSpeech/gs_dev_shards.list \
	--checkpoint $decode_checkpoint \
	--beam_size 8 \
	--batch_size 4 \
	--length_penalty 0.0 \
	--result_dir $result_dir \
	--reverse_weight $reverse_weight \
	--ctc_weight $ctc_weight \
	--transducer_weight $transducer_weight

	python tools/compute-wer.py --char=1 -v=1 /shared/speech-shards/dev/en/GigaSpeech/text $result_dir/$mode/text > $result_dir/$mode/results.wer
	tail $result_dir/$mode/results.wer
	python tools/compute-wer-giga.py --char=1 -v=1 /shared/speech-shards/dev/en/GigaSpeech/text $result_dir/$mode/text > $result_dir/$mode/results.filtered.wer
	tail $result_dir/$mode/results.filtered.wer
done