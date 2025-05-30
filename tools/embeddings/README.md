
# Embeddings Tools


## Creating Embeddings
`get_embeddings.py`: takes an ASR model and input data and produces a directory with files: `sample_names`, `embeddings_layer_0`, `embeddings_layer_1`, ..., `embeddings_layer_n`. 

The `j`th row in `embeddings_layer_i` is an audio embedding of the segment `sample_names[j]` taken from the `i`th layer of the encoder. 

Required Inputs:
- `--config`: ASR model config
- `--test_data`: a list of shards or a list of raw segments
- `--checkpoint`: ASR model checkpoint
- `--output_name`: name of directory where `sample_names` and `embeddings_layer_i` files are written


## Plotting Embeddings

`plot_embeddings.py`: is a tool to help explore our audio embeddings. This script downloads and streams files from `s3://nerd-2931`, and requires:
1. the output files from `get_embeddings.py` to be uploaded to `s3://nerd-2931/<directory>/embeddings/`
2. the wav file segments to be uploaded to `s3://nerd-2931/<directory>/audios/` (the names of the wav files should match the names in `sample_names`)

Inputs:
- `--layer_id`: Specify which layer's embeddings to use (default is `1`)
- `--data_sets`: names of folders in `s3://nerd-2931/` to use. Currently available: `autotc`, `earnings21`, `earnings22`, `luminary`, `axon_202005`, `axon_2024`, `deletions`.