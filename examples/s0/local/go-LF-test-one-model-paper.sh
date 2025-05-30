#!/bin/bash


# Function to clean up background processes
cleanup() {
  echo "Terminating background processes..."
  kill 0 # Kills all processes in the current process group
  wait   # Wait for all background processes to finish
}

# Trap SIGINT (Ctrl-C) and call cleanup
trap cleanup SIGINT

function run_testset {
    base_cfg=$1
    nickname=$2
    decode_mode=$3
    gpu=$4
    chunk_size=$5
    batch_size=$6
    beam_size=$7
    xcs=$(printf "%06d" $chunk_size)
    extras=$8
    
    yaml_config=/shared/experiments/paper_replacing_MHA/models/$base_cfg.run-config.yaml
    checkpoint=/shared/experiments/paper_replacing_MHA/models/$base_cfg.pt
    
    result_dir=results.$pnick/test/$nickname-b$batch_size.bs$beam_size.cs$xcs

    if [ -e "$result_dir/results/summary.txt" ]; then
            echo "summary.txt already exists for $result_dir, skipping"
            return 0
    fi

    for f in /shared/speech-shards/test/en/Gigaspeech_long/wav/*.wav; do
	x=$(basename $f .wav)
	if [ -e "$result_dir/$decode_mode/$x.ctm" ]; then
	    echo "$x.ctm already exists in $result_dir/$decode_mode/$x.ctm, skipping"
        else
	   echo "$x.ctm does not exist in $result_dir/$decode_mode/$x.ctm, decoding"

        python wenet/bin/recognize_wav2.py $extras \
            --config $yaml_config \
            --checkpoint $checkpoint \
            --modes $decode_mode \
            --batch_size $batch_size \
	    --gpu $gpu \
            --audio_file $f \
            --result_dir $result_dir \
            --chunk_size $chunk_size \
            --beam_size $beam_size \
            --ctc_weight 0.3 \
            --transducer_weight 0.7 \
            --reverse_weight 0.5
	fi
    done
    
    python local/gigaspeech_scoring_longform.py '/shared/speech-shards/test/en/Gigaspeech_long/*.ref_txt' "$result_dir/$decode_mode/*ctm" &


    
    rwkvEnv=$(env | grep RWKV)

    cat << EOF | tee $result_dir/runinfo.txt
Metadata: 
nickname: $nickname
config: $yaml_config
checkpoint : $checkpoint
mode: $decode_mode
gpu: $gpu
chunk_size: $chunk_size
beam_size : $beam_size
batch_size : $batch_size
extras: $extras
rwkv_env: $rwkvEnv

EOF
}


export AA=$1
export BB=$2
extras=""
g=${3:-0}
decode_mode=${4:-rnnt_beam_search}
pnick=${5:-paper}
echo "decode-mode $decode_mode, pnick $pnick"
echo "Model=$AA"
echo "Nickname=$BB"

batch_size=8
run_testset $AA $BB rnnt_beam_search $g  2000 16 8  "$extras" &
sleep 2

batch_size=16
chunk_size=4000
g=$(( g + 1 ))
run_testset $AA $BB $decode_mode $g $chunk_size $batch_size 8 "$extras" &
sleep 2

batch_size=8
chunk_size=9000
g=$(( g + 1 ))
run_testset $AA $BB $decode_mode $g $chunk_size $batch_size 8 "$extras" &
sleep 2

chunk_size=15000
g=$(( g + 1 ))
run_testset $AA $BB $decode_mode $g $chunk_size $batch_size 8 "$extras" &
sleep 2


batch_size=8
chunk_size=20000
g=$(( g + 1 ))
run_testset $AA $BB $decode_mode $g $chunk_size $batch_size 8 "$extras" &
sleep 2

batch_size=8
chunk_size=40000
g=$(( g + 1 ))
run_testset $AA $BB $decode_mode $g $chunk_size $batch_size 8 "$extras" &
sleep 2

wait
exit
