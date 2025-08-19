# Paper Citation

**"Accurate, fast, cheap: Choose three. Replacing Multi-Head-Attention with Bidirectional Recurrent Attention for Long-Form ASR"**  
*Martin Ratajczak, Jean-Philippe Robichaud, Jennifer Drexler Fox*  
Interspeech 2025

**Links:**
- [ArXiv Paper](https://arxiv.org/abs/2506.19761)
- [Interspeech 2025 Proceedings](https://www.isca-archive.org/interspeech_2025/ratajczak25_interspeech.html)
- **Interspeech Talk**: Wednesday, August 20, 2025, 9:10-9:30 AM

**Citation:**
```bibtex
@inproceedings{ratajczak25_interspeech,
  title     = {{Accurate, fast, cheap: Choose three. Replacing Multi-Head-Attention with Bidirectional Recurrent Attention for Long-Form ASR}},
  author    = {{Martin Ratajczak and Jean-Philippe Robichaud and Jennifer {Drexler Fox}}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{3324--3328}},
  doi       = {{10.21437/Interspeech.2025-2059}},
  issn      = {{2958-1796}},
}
```


# Setup

 * conda & wenet `requirements.txt`, follow regular wenet 
 * use `. path.sh` once the conda/micromamba environment is activated.
 * compile and install the `optimized_transducer` loss from [optimized_transducer](https://github.com/csukuangfj/optimized_transducer)
 * install and compile `fstalign`, follow instructions from the repository

Note: for mamba, we had slight modifications done in their repository in order to wrap the bidirectional 
      configuration.  We will submit them to the original project repository.


``` sh
micromamba create -n wenet python=3.10
micromamba activate wenet
micromamba install conda-forge::sox
micromamba install pytorch=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
cd  paper_accurate_fast_cheap # The root of this repository
pip install -r requirements.txt

```

# Data
  Follow regular WeNet GigaSpeech recipe to prepare the data into segmented shards.  
Also, keep track of dev and test full audios and convert them to regular 16kHz PCM wav files.

# Training

## Short-Form base experiments

1. baseline MHA conformer : `conf/conformer/giga.conformer_ds4k31nc_12le.trans.shortform.cfg`
1. mamba-2 : 
    1. mamba-2, uni       : `conf/mamba/giga.mamba_ds4k31nc_12le.trans.shortform.cfg`
    1. mamba-2, bi        : `conf/mamba/giga.mambabi_ds4k31nc_12le.trans.shortform.cfg`
1. RWKV:
    1. rwkv, uni          : `conf/rwkv/giga.rwkv_ds4k31nc_12le.trans.shortform.cfg`
    1. rwkv, bi           : `conf/rwkv/giga.rwkvbi_ds4k31nc_12le.trans.shortform.cfg`
    1. rwkv, DirDrop-R2L  : `conf/rwkv/giga.rwkvdld_ds4k31nc_12le.trans.shortform.cfg`
    1. rwkv, DirDrop-Both : `giga.rwkvbi_dldb_ds4k31nc_12le.trans.shortform.cfg`

## Long-Form base experiments

1. MHA
    1. baseline MHA conformer          : `conf/conformer/giga.conformer_ds4k31nc_12le.trans.longform.cfg`
    1. MHA + LCA256+GT first fine-tune : `conf/conformer/giga.conformer_ds4k31nc_12le.trans.FT-LF-LA256-GT.cfg`
    1. MHA + LCA256+GT + FT-LFXL       : `conf/conformer/giga.conformer_ds4k31nc_12le.LCA256-GT-FT-LFXL.max.cfg`
1. RWKV:
    1. rwkv, uni          : `conf/rwkv/giga.rwkv_uni_ds4k31nc_12le.trans-longutts.cfg`
    1. rwkv, uni, FT-LFXL : `conf/rwkv/giga.rwkvuni_ds4k31nc_12le.FT-LFXL.max.cfg`
    1. rwkv, bi           : `conf/rwkv/giga.rwkvbi_ds4k31nc_12le.trans-longutts.cfg`
    1. rwkv, bi, FT-LFXL  : `conf/rwkv/giga.rwkvbi_ds4k31nc_12le.FT-LFXL.max.cfg`


# Decoding


The scripts below assume that each model checkpoint has an accompanying ".run-config.yaml" script
(that can be different from the yaml file used for training).

For example, we expect that an average checkpoint like `giga.conformer_ds4k31nc_12le.trans.LF.avg10_at_ep44.pt` will 
be sitting next to `giga.conformer_ds4k31nc_12le.trans.LF.avg10_at_ep44.run-config.yaml`.  
In the text below, we'll call it `giga.conformer_ds4k31nc_12le.trans.LF.avg10_at_ep44` the "basename" of the model checkpoint.


# Short-Form WER


### Regular SF decoding using segmented shards


```
# source the environment
bash local/go-SF-dev-one-model-paper.sh /full/path/to/checkpoint.pt output_result_dir
grep Overall output_result_dir/rnnt_beam_search/results.filtered.wer
```

The two useful scripts 
* `local/go-SF-dev-one-model-paper.sh`  : run evaluation on the devset 
* `local/go-SF-test-one-model-paper.sh`  : run evaluation on the testset 


### Instructions for alternate-decoding and disabling bidirectional layers

When a bidirectional was trained with dropout mode, you can also use alternate-decoding and
you can also decide to activate/deactivate any of the R2L layers. See the paper (or the code!) for a description
of the alternate-decoding.

To turn on alternate-decoding, set the following shell variable:
* `RWKV_ALT_DECODING=1 bash local/go-SF-dev-one-model-paper.sh </full/path/to/checkpoint.pt> <output_result_dir> <gpuid>`

If the `RWKV_BIDIRECTIONAL_LAYERS` environment variable exists, its content will control which "bidirectional" layers
are active (which layers have both the L2R and R2L recurrent attention block enabled. Only the layers
whose id match a number in that list will have bidirectional enabled.  Otherwise, only one direction will be enabled. 

Layer-id starts at 0. Here's a shorthand of configurations:
* `RWKV_BIDIRECTIONAL_LAYERS="-1"` # none of the layers are forced in bidirectional mode since none have a layer ID of -1.
* `RWKV_BIDIRECTIONAL_LAYERS="0"` # only the first layer of the model will have its bidirectional layer enabled.
* `RWKV_BIDIRECTIONAL_LAYERS="11"` # only the last layer of a 12-Layer model will have its bidirectional layer enabled.
* `RWKV_BIDIRECTIONAL_LAYERS="9,10,11"` # only the last three layers of a 12-Layer model will have its bidirectional layer enabled.
* `RWKV_BIDIRECTIONAL_LAYERS="6,7,8,9,10,11"` # only the last six layers of a 12-Layer model will have its bidirectional layer enabled.

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
bash go-LF-dev-one-model-paper.sh <model_checkpoint_basename> <results_folder> <gpuid>   # running on the dev set
bash go-LF-test-one-model-paper.sh <model_checkpoint_basename> <results_folder> <gpuid>   # running on the test set
```

Each of these will run, in sequence, the decoding on various chunk sizes.  
Look for `summary.txt` for the final aggregated accuracy results. 
Note that you will likely need to adjust the paths in these scripts to match your own configuration.


## Encoder RTF

The original audio file we used here can't be released, so we suggest you create another long wav file using sox
by concatenating a single file multiple times. Refer to that file in the `go-run-encoder-rtf.single-gpu-3x3-g5.sh` script.

```
bash local/go-run-encoder-rtf.single-gpu-3x3-g5.sh
python tools/rtf/get-rtf-tables.py results.encoder-rtf
```


