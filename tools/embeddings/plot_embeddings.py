from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import boto3
import requests
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import os
import argparse

# get aws credentials from environment 
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = os.environ.get('AWS_SESSION_TOKEN')

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                  aws_session_token=AWS_SESSION_TOKEN,
                  region_name='us-west-2')

matplotlib.use("Qt5Agg") # for interactive stuff

def parse_args():
    parser = argparse.ArgumentParser("Download embeddings from S3 and make an interactive plot")
    parser.add_argument('--layer_id', help='plot embedding vectors produced from this layer', default=1)
    parser.add_argument('--data_sets', help='plot embeddings from data at s3://nerd-2931/<data_set>', default=['axon_202005', 'autotc', 'earnings21', 'luminary'], nargs='+')
    parser.add_argument('--use-small-sample', help='use sample of 1000 embeddings from each test suite', type=bool, default=False)
    args = parser.parse_args()
    return args

def load_file_from_s3(s3_key):
    # check if file is in cache
    if os.path.exists(f".s3_cache/{s3_key}"):
        print(f"Using cached embeddings at .s3_cache/{s3_key}")
    else: 
        os.makedirs(f".s3_cache/{s3_key.split('/')[0]}/embeddings", exist_ok=True)
        try:
            print(f"downloading s3://nerd-2931/{s3_key} to .s3_cache/{s3_key}")
            response = s3.download_file('nerd-2931', s3_key, f".s3_cache/{s3_key}")
        except Exception as e:
            print(f"Error downloading embeddings from S3. \n Are your AWS credentials set in your environment? \n Do your data_sets exist? \n {e}")
            exit(1)
    return f".s3_cache/{s3_key}"


def load_embeddings(folder_names, layer_id, use_small_sample):
    embeddings = defaultdict(list)
    for folder_name in folder_names:
        # get sample_names
        if use_small_sample:
            s3_key = f"{folder_name}/embeddings/sample_names_1000"
        else:
            s3_key = f"{folder_name}/embeddings/sample_names"
        with open(load_file_from_s3(s3_key)) as f:
            sample_names = [name.strip() for name in f.readlines()]

        # get embeddings
        if use_small_sample:
            s3_key = f"{folder_name}/embeddings/embeddings_layer_{layer_id}_1000"
        else:
            s3_key = f"{folder_name}/embeddings/embeddings_layer_{layer_id}"
        with open(load_file_from_s3(s3_key)) as f:
            data = np.loadtxt(f, delimiter=',')
        data_list = [data[i] for i in range(data.shape[0])]
        assert len(sample_names) == len(data_list), f"embeddings file {s3_key} has {len(data_list)} entries instead of {len(sample_names)}"
        embeddings[f'layer_{layer_id}'] += data_list
        embeddings['class_names'] += [folder_name] * len(sample_names)
        embeddings['sample_names'] += sample_names
    
    df = pd.DataFrame(embeddings)
    return df


def do_tsne(layer_id, df):                                     
    assert f'layer_{layer_id}' in df.columns, f"layer id {layer_id} not in dataframe"

    X = np.stack(df[f'layer_{layer_id}'].tolist(), axis=0)
    tsne = TSNE(random_state=0, n_iter=1000)
    tsne_results = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df_tsne['class'] = df['class_names'].tolist()
    df_tsne['sample_names'] = df['sample_names'].tolist()
    return df_tsne


def get_s3_presigned_url(s3_key):
    # Generate a presigned URL for the S3 object in bucket nerd-2931
    response = s3.generate_presigned_url('get_object',
                                         Params={'Bucket': "nerd-2931", 'Key': s3_key},
                                         ExpiresIn=3600)
    return response


def stream_audio_from_s3(s3_key):
    # play audio stored on S3
    url = get_s3_presigned_url(s3_key)
    response = requests.get(url, stream=True)
    audio_data = BytesIO(response.content)
    try:
        audio = AudioSegment.from_file(audio_data)
        print(f"Playing audio file s3://nerd-2931/{s3_key}")
        play(audio)
    except Exception as e:
        print(f"Trouble playing audio. Refresh your AWS credentials? {e}")
    

def plot_embeddings_interactive(layer_id, df, folder_names):
    df_tsne = do_tsne(layer_id, df)
    print("total number of points: ", len(df_tsne))
    fig, ax = plt.subplots(figsize=(8,6)) # Set figsize
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'purple', 'orange']
    color_map = {fn: color for fn, color in zip(folder_names, colors)}
    sc = ax.scatter(df_tsne['TSNE1'].tolist(), df_tsne['TSNE2'].tolist(), color=df_tsne['class'].apply(lambda x: color_map[x]))
    
    custom_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=class_name) for class_name, color in zip(folder_names, colors)]
    plt.legend(handles=custom_legend, loc='upper left')
    
    plt.title(f'encoder {layer_id} embeddings using t-SNE');
    plt.xlabel('TSNE1');
    plt.ylabel('TSNE2');
    plt.axis('equal')
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        # get closest segment name and write on plot
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = ' '.join(str(df_tsne['sample_names'].iloc[int(i)]) for i in ind['ind'])
        annot.set_text(text)
    
    def hover(event):
        # annotate segment name on hover
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            elif vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    def on_press(event):
        # play audio of nearest segment on mouse click
        cont, ind = sc.contains(event)
        if cont:
            df_index = ind["ind"][0]
            segment_name = df_tsne['sample_names'].iloc[int(df_index)].strip()
            class_name = df_tsne['class'].iloc[int(df_index)].strip()
            s3_key = f'{class_name}/audios/{segment_name}.wav'
            stream_audio_from_s3(s3_key)

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect('button_press_event', on_press)
    
    plt.show()


def main(args):
    df = load_embeddings(args.data_sets, args.layer_id, args.use_small_sample)
    plot_embeddings_interactive(args.layer_id, df, args.data_sets)


if __name__=='__main__':
    args = parse_args()
    print("use small sample: ", args.use_small_sample)
    main(args)
