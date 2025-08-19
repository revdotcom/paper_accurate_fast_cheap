
Rev.com
Notes from the paper  Accurate, fast, cheap: choose three.


# Setup

 * conda & wenet requirements.txt, follow regular wenet 
 * use `. path.sh` once the conda/micromamba environment is activated.
 * install and compile fstalign, follow instructions from the repository

Note: for mamba, we had slight modifications done in their repository in order to wrap the bidirectional 
      configuration.  We will submit them to the original project repository.

# Data
  Follow regular WeNet GigaSpeech recipe to prepare the data into segmented shards.  
Also, keep track of dev and test full audios and convert them to regular 16kHz PCM wav files.

# Training

## Short-Form base experiments

** Changed "martin" to "shortform" everywhere

1. baseline MHA conformer : conf/conformer/giga.conformer_ds4k31nc_12le.trans.shortform.cfg
1. mamba-2 : 
    1. mamba-2, uni       : conf/mamba/giga.mamba_ds4k31nc_12le.trans.shortform.cfg
    1. mamba-2, bi        : conf/mamba/giga.mambabi_ds4k31nc_12le.trans.shortform.cfg
1. RWKV:
    1. rwkv, uni          : conf/rwkv/giga.rwkv_ds4k31nc_12le.trans.shortform.cfg
    1. rwkv, bi           : conf/rwkv/giga.rwkvbi_ds4k31nc_12le.trans.shortform.cfg
    1. rwkv, DirDrop-R2L  : conf/rwkv/giga.rwkvdld_ds4k31nc_12le.trans.shortform.cfg
    1. rwkv, DirDrop-Both : giga.rwkvbi_dldb_ds4k31nc_12le.trans.shortform.cfg

## Long-Form base experiments

1. MHA
    1. baseline MHA conformer          : conf/conformer/giga.conformer_ds4k31nc_12le.trans.longform.cfg
    1. MHA + LCA256+GT first fine-tune : conf/conformer/giga.conformer_ds4k31nc_12le.trans.FT-LF-LA256-GT.cfg
    1. MHA + LCA256+GT + FT-LFXL       : conf/conformer/giga.conformer_ds4k31nc_12le.LCA256-GT-FT-LFXL.max.cfg
1. RWKV:
    1. rwkv, uni          : conf/rwkv/giga.rwkv_uni_ds4k31nc_12le.trans-longutts.cfg
    1. rwkv, uni, FT-LFXL : conf/rwkv/giga.rwkvuni_ds4k31nc_12le.FT-LFXL.max.cfg
    1. rwkv, bi           : conf/rwkv/giga.rwkvbi_ds4k31nc_12le.trans-longutts.cfg
    1. rwkv, bi, FT-LFXL  : conf/rwkv/giga.rwkvbi_ds4k31nc_12le.FT-LFXL.max.cfg


# Decoding


The scripts below assume that each model checkpoint has an accompagning ".run-config.yaml" script
(that can be different from the yaml file used for training).

For example, we expect that a an average checkpoint like `giga.conformer_ds4k31nc_12le.trans.LF.avg10_at_ep44.pt` will 
be sitting next to `giga.conformer_ds4k31nc_12le.trans.LF.avg10_at_ep44.run-config.yaml`.  
In the text below, we'll calli `giga.conformer_ds4k31nc_12le.trans.LF.avg10_at_ep44` the "basename" of the model checkpoint.


# Short-Form WER


### Regular SF decoding using segmented shards


```
# source the environment
bash local/go-SF-dev-one-model-paper.sh /full/path/to/checkpoint.pt output_result_dir
grep Overall output_result_dir/rnnt_beam_search/results.filtered.wer
```

The two useful scripts 
* local/go-SF-dev-one-model-paper.sh  : run evaluation on the devset 
* local/go-SF-test-one-model-paper.sh  : run evaluation on the devset 


### Instructions for alternate-decoding and disabling bidirectional layers

When a bidirectional was trained with dropout mode, you can also use alternate-decoding and
you can also decide to activate/deactivate any of the the R2L layers. See the paper (or the code!) for a description
of the alternate-decoding.

To turn on simply alternate-decoding, simply set this the following shell variable:
* `RWKV_ALT_DECODING=1 bash local/go-SF-dev-one-model-paper.sh </full/path/to/checkpoint.pt> <output_result_dir> <gpuid>`

If the `RWKV_BIDIRECTIONAL_LAYERS` environment variable exists, its content will control which "bidirectional" layers
are active (which layers have both the L2R and R2L recurrent attention block enabled. Only the layers
whose id match a number in that list will have bidirectional enabled.  Otherwise, only one direction will be enabled. 

Layer-id starts at 0. Here's a shortand of configurations:
* `RWKV_BIDIRECTIONAL_LAYERS="-1"` # none of the layers are forced in bidirectional mode since none have a layerid of -1.
* `RWKV_BIDIRECTIONAL_LAYERS="0"` # only the first layer of the model will have its bidirectional layer enabled.
* `RWKV_BIDIRECTIONAL_LAYERS="11"` # only the last layer of a 12-Layer model will have its bidirectional layer enabled.
* `RWKV_BIDIRECTIONAL_LAYERS="9,10,11"` # only the last three layers of a 12-Layer model will have its bidirectional layer enabled.
* `RWKV_BIDIRECTIONAL_LAYERS="6,7,8,9,10,11"` # only the last three layers of a 12-Layer model will have its bidirectional layer enabled.

Both of `RWKV_ALT_DECODING` and `RWKV_BIDIRECTIONAL_LAYERS` can be active at the same time:

* `RWKV_ALT_DECODING=0 RWKV_BIDIRECTIONAL_LAYERS="9,10,11" bash local/go-SF-dev-one-model-paper.sh /full/path/to/checkpoint.pt output_result_dir_1` will have regular decoding with a restriction on the bidirectional layers.  Layers 0 to 8 will have only the L2R recurrent attention direction enabled, and the last 3 layers will be in full bidirectional mode.
* `RWKV_ALT_DECODING=1 RWKV_BIDIRECTIONAL_LAYERS="9,10,11" bash local/go-SF-dev-one-model-paper.sh /full/path/to/checkpoint.pt output_result_dir_1` will have regular decoding with a restriction on the bidirectional layers.  Layers 0,2,4,6 and 8 will have only the L2R recurrent attention direction enabled, layers 1,3,5,7 will have only the R2L recurrent attention enabled and the last 3 layers will be in full bidirectional mode.



## Long-Form WER

Here, we use a different scoring script, which requires the availability of fstalign.  Also, you need to generate the full utterances
text transcription:

```
cat /path/to/dev/GigaSpeech/text | python local/segments_to_files.py /path/to/dev/GigaSpeech_long/
```
This will create one `.ref_txt` file for each full audio file as defined in the GigaSpeech `segments` file.


To run an experiment in long-form:

```
bash go-LF-dev-one-model-paper.sh <model_checkpoint_basename> <results_folder> <gpuid>   # rnning on the dev set
bash go-LF-test-one-model-paper.sh <model_checkpoint_basename> <results_folder> <gpuid>   # rnning on the test set
```

Each of these will run, in sequence, the decoding on various chunk sizes.  
Look for "summary.txt" for the final aggregated accuracy results. 
Note that you will likely need to adjust the paths in these scripts to match your own configuration.


## Encoder RTF

The original audio file we used here can't be released, so we suggest you create another long wav file using sox
by concatenating a single file multuple time. Refer to that file in the `go-run-encoder-rtf.single-gpu-3x3-g5.sh` script.

```
bash local/go-run-encoder-rtf.single-gpu-3x3-g5.sh
python tools/rtf/get-rtf-tables.py results.encoder-rtf
```


