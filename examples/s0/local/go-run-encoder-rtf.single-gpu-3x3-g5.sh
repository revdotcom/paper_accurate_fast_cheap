#!/bin/bash

decode_mode=rnnt_beam_search
run=00

function run_enc_rtf {
    base_cfg=$1
    nickname=$2
    decode_mode=$3 # ignored
    gpu=$4
    chunk_size=$5
    batch_size=$6
    beam_size=$7
    extra="$8"
    
    yaml_config=/shared/experiments/paper_replacing_MHA/models/$base_cfg.run-config.yaml
    checkpoint=/shared/experiments/paper_replacing_MHA/models/$base_cfg.pt

    xcs=$(printf "%06d" $chunk_size)
    
    result_dir=results.encoder-rtf/runs/$nickname-cs$xcs.run$run.bs$batch_size.gpu

    # forcing this here
    gpu=0

    
    if [ -e "$result_dir/rtf/file3x3_9857s.rtf" ]; then
	    echo "rtf result already exists for $result_dir, skipping"
            echo "---"
	    return 0
    fi
    
    python wenet/bin/encoder-rtf.py $extra \
        --audio_file NERD-3065/audios/file3x3_9857s.wav \
        --checkpoint $checkpoint \
        --config $yaml_config \
        --chunk_size $chunk_size \
        --batch_size $batch_size \
        --result_dir $result_dir \
        --gpu $gpu \
        --warmup 3

    cat << EOF  >  $result_dir/metadata.txt
Metadata: 
run        : $run
nickname   : $nickname
config     : $yaml_config
checkpoint : $checkpoint
mode       : $decode_mode
gpu        : $gpu
chunk_size : $chunk_size
beam_size  : $beam_size
batch_size : $batch_size
EOF

}


for run in 01 02 03 04 05; do
for batch_size in 4 8 1 10 12 14; do
for chunk_size in 2000 4000 9000 15000 20000 40000 60000 100000 200000 ; do


 run_enc_rtf giga.rwkv_uni_ds4k31nc_12le.trans-LF.swept-snowball-3.avg10_at_0060kh rwkv_uni_12L rnnt_beam_search 5 $chunk_size $batch_size 8
 run_enc_rtf giga.rwkv_ds4k31nc_18le.trans.martin.2_ep33_at740000_sn1              rwkv_uni_18L rnnt_beam_search 3 $chunk_size $batch_size 8

 run_enc_rtf giga.rwkvbi_ds4k31nc_12le.trans.martin.1.bright-puddle-2.avg10_at_0240kh rwkv_bi_12L rnnt_beam_search 2 $chunk_size $batch_size 8
 run_enc_rtf giga.rwkvbi_ds4k31nc_18le.trans.martin.1_avgsn10_ep30_at670000_sn1       rwkv_bi_18L rnnt_beam_search 2 $chunk_size $batch_size 8
 run_enc_rtf giga.rwkvbi_ds4k31nc_24le.trans.martin.1_ep40_avg10_at756000_sn1         rwkv_bi_24L rnnt_beam_search 4 $chunk_size $batch_size 8
 run_enc_rtf giga.rwkvbi_ds4k31nc_30le.trans.martin.1_ep46_at626000_sn1               rwkv_bi_30L rnnt_beam_search 5 $chunk_size $batch_size 8


 run_enc_rtf giga.conformer_ds4k31nc_12le.trans.martin.1_ep46_avgsn10_at1050000_sn1       mha_12L rnnt_beam_search 6 $chunk_size $batch_size 8  
 run_enc_rtf giga.conformer_ds4k31nc_18le.trans.martin.1.glowing-dragon-3.avg10_at_0420kh mha_18L rnnt_beam_search 0 $chunk_size $batch_size 8
 run_enc_rtf giga.conformer_ds4k31nc_24le.trans.fake.astral-cosmos-2.avg10_at_0020kh      mha_24L rnnt_beam_search 0 $chunk_size $batch_size 8
 run_enc_rtf giga.conformer_ds4k31nc_30le.trans.fake.wild-night-1.avg02_at_0030kh         mha_30L rnnt_beam_search 0 $chunk_size $batch_size 8

 run_enc_rtf giga.mamba2bi_ds4k31nc_12le.martin.1_avgsnap_20_at_1420000_ep59_1 mamba2bi_12L  rnnt_beam_search 5 $chunk_size $batch_size 8
 run_enc_rtf giga.mamba_ds4k31nc_12le.trans.martin.1.floral-feather-7.avg10_at_0430kh mamba2_uni_12L rnnt_beam_search 5 $chunk_size $batch_size 8
 run_enc_rtf giga.conformer_ds4k31nc_12le.trans.FT-LF-LA256-GT.gallant-bush-4.avg01_at_0004kh mha_12L-LA256-GT  rnnt_beam_search 5 $chunk_size $batch_size 8
 
 run_enc_rtf giga.rwkvbi_ds4k31nc_12le.LFXL.wandering-forest-3.avg10_at_0100kh rwkvbi_12L_LFXL  rnnt_beam_search 5 $chunk_size $batch_size 8
 
 export RRWKV_ALT_DECODING=1
 export RWKV_BIDIRECTIONAL_LAYERS="-1"
 run_enc_rtf giga.rwkvbi_dldb_ds4k31nc_12le.trans.martin.1_ep44_avg30_minstep956000 rwkvbi_12L_alt-only  rnnt_beam_search 5 $chunk_size $batch_size 8
 
 export RRWKV_ALT_DECODING=1
 export RWKV_BIDIRECTIONAL_LAYERS="11"
 run_enc_rtf giga.rwkvbi_dldb_ds4k31nc_12le.trans.martin.1_ep44_avg30_minstep956000 rwkvbi_12L_alt-bi11  rnnt_beam_search 5 $chunk_size $batch_size 8
 
 export RRWKV_ALT_DECODING=1
 export RWKV_BIDIRECTIONAL_LAYERS="9,10,11"
 run_enc_rtf giga.rwkvbi_dldb_ds4k31nc_12le.trans.martin.1_ep44_avg30_minstep956000 rwkvbi_12L_alt-bi9-11  rnnt_beam_search 5 $chunk_size $batch_size 8
 
 
 export RRWKV_ALT_DECODING=1
 export RWKV_BIDIRECTIONAL_LAYERS="0"
 run_enc_rtf giga.rwkvbi_dldb_ds4k31nc_12le.trans.martin.1_ep44_avg30_minstep956000 rwkvbi_12L_alt-BiFirst  rnnt_beam_search 5 $chunk_size $batch_size 8
 
 export RRWKV_ALT_DECODING=1
 export RWKV_BIDIRECTIONAL_LAYERS="6,7,8,9,10,11"
 run_enc_rtf giga.rwkvbi_dldb_ds4k31nc_12le.trans.martin.1_ep44_avg30_minstep956000 rwkvbi_12L_alt-BiLast6  rnnt_beam_search 5 $chunk_size $batch_size 8

done
done
done


