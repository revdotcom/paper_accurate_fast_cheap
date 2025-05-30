import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import os
import itertools
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

global max_minutes_per_sec 
max_minutes_per_sec = 0

base_dir = "NERD-3065/results.cold_start"
base_dir = "NERD-3065/results.warmup2.cpu"
base_dir = "NERD-3065/results.warmup2" 
base_dir = "NERD-3065/results.warmup2.file3" 
base_dir = "results.encoder-rtf"

if len(sys.argv) > 1:
  base_dir = sys.argv[1]

graph_dir = f"{base_dir}/graphs"
glob_pattern = f"{base_dir}/runs/*/rtf/*.rtf"
os.makedirs(graph_dir + "/runs", exist_ok=True)
files_to_process = glob.glob(glob_pattern)
print(list(files_to_process))
# sys.exit(0)

# file names will be of NERD-3065/results/<configuration>/rtf/<file>.rtf
#configuration will look like :
# reverb_v1-cs010000.run01.bs4.gpu
# TS-DEI-01-cs012000.run04.bs6.gpu
# where:
# reverb_v1 or TS-DEI-01 are examples of model names (more names than this will be found in the list of files)
# cs<0-padded numbers> is the decoding chunk size
# run<0-padded numbers> is the run number, there for reproducibility
# bs<number> is the batch size used for deocding 
# gpu|cpu will tell us if the decoding was done on GPU or CPU

# The content of a file will look like:
# num_frames=   40000, elapsed=0.819277, local_rtf=0.002048, 8.14 minutes of audio processed per sec
# num_frames=   40000, elapsed=0.088956, local_rtf=0.000222, 74.94 minutes of audio processed per sec
# num_frames=   36136, elapsed=0.088910, local_rtf=0.000246, 67.74 minutes of audio processed per sec
# total_frames=  116136, total_elapsed=0.997143, final_rtf 0.000859, 19.41 minutes of audio processed per sec
# max_vram=6858.09 MB, max_cpu_ram=2730.36 MB
# some files will be empty

def parse_file(file_path):
    global max_minutes_per_sec 
    with open(file_path, 'r') as f:
        content = f.read()
    
    rtf_match = re.search(r'final_rtf (\d+\.\d+)', content)
    vram_match = re.search(r'max_vram=(\d+\.\d+)', content)
    ram_match = re.search(r'max_cpu_ram=(\d+\.\d+)', content)
    #minutes_match = re.search(r'(\d+\.\d+) minutes of audio processed per sec', content)
    minutes_match = re.search(r'final_rtf \d+\.\d+,\s+(\d+\.\d+) minutes of audio processed per sec', content, re.M)
    
    if rtf_match and vram_match and ram_match and minutes_match:
        mps = float(minutes_match.group(1))
        if mps > max_minutes_per_sec:
           max_minutes_per_sec = mps
        return {
            'rtf': float(rtf_match.group(1)),
            'vram': float(vram_match.group(1)),
            'ram': float(ram_match.group(1)),
            'minutes_per_sec': mps
        }
    return None

def parse_config(file_path):
    parts = file_path.split('/')
    config = parts[-3]
    # print(f"{config=}")
    model, rest = config.split('-cs')
    chunk_size = int(rest.split('.')[0])
    batch_size = int(re.search(r'bs(\d+)', config).group(1))
    device = 'GPU' if 'gpu' in config else 'CPU'
    model = model + "-" + device
    return model, chunk_size, batch_size, device

# Modify the data structure to include batch size
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

valid_files = []
for file in tqdm(list(files_to_process), desc="Results files processed"):
    try:
        result = parse_file(file)
        # print(f"from {file} we got {result=}")
        if result:
            model, chunk_size, batch_size, device = parse_config(file)
            data[model][chunk_size][batch_size].append(result)
            valid_files.append(file)
    except Exception as ex:
        print(f"skipping {file} as it couldn't be parsed")
        print(ex)

files_to_process = valid_files

# Define a list of markers to cycle through
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
marker_cycle = itertools.cycle(markers)

def plot_metric(metric_name, ylabel, batch_size=1):
    plt.figure(figsize=(14, 8))
    marker_dict = {}
    for model in data:
        x = []
        y = []
        yerr = []
        for chunk_size in sorted(data[model]):
            # values = [r[metric_name] for batch_size in data[model][chunk_size] for r in data[model][chunk_size][batch_size]]
            values = [r[metric_name] for r in data[model][chunk_size][batch_size]]
            if values:
                x.append(chunk_size)
                y.append(np.mean(values))
                yerr.append(np.std(values))
        if x:  # Only plot and add to legend if there's data
            if model not in marker_dict:
                marker_dict[model] = next(marker_cycle)
            plt.errorbar(x, y, yerr=yerr, label=model, capsize=5, marker=marker_dict[model])
    
    #plt.xscale('log')
    plt.xlabel('Chunk Size')
    plt.ylabel(ylabel)
    if metric_name == "minutes_per_sec":
        plt.ylim(0, max_minutes_per_sec)
    plt.title(f'{ylabel} vs Chunk Size (batch size of {batch_size})')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{graph_dir}/{metric_name}_vs_chunk_size_bs{batch_size}.png', bbox_inches='tight')
    plt.close()

def plot_batch_size_effect(metric_name, ylabel, restrict_model=None):
    plt.figure(figsize=(14, 8))
    marker_dict = {}
    for model in data:
        if restrict_model is not None and restrict_model != model:
            continue
        batch_sizes = set(batch_size for chunk_size in data[model] for batch_size in data[model][chunk_size])
        chunk_sizes = sorted(data[model])
        print(f"Model: {model}, Batch sizes: {batch_sizes}, Chunk sizes {chunk_sizes}")  # Debugging line
        
        for batch_size in sorted(batch_sizes):
            x = []
            y = []
            yerr = []
            for chunk_size in sorted(data[model]):
                values = [r[metric_name] for r in data[model][chunk_size].get(batch_size, [])]
                if values:
                    x.append(chunk_size)
                    y.append(np.mean(values))
                    yerr.append(np.std(values))
            if x:  # Only plot and add to legend if there's data
                if model not in marker_dict:
                    marker_dict[model] = next(marker_cycle)
                plt.errorbar(x, y, yerr=yerr, label=f'{model} (BS={batch_size})', capsize=5, marker=marker_dict[model])
    
    #plt.xscale('log')
    plt.xlabel('Chunk Size')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs Chunk Size (Batch Size Effect)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if restrict_model is not None:
        plot_filename = f"{graph_dir}/{restrict_model}_{metric_name}_vs_chunk_size_batch_effect.png"
    else:
        plot_filename = f"{graph_dir}/{metric_name}_vs_chunk_size_batch_effect.png"
    print(f"Saving {plot_filename}")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

def create_plotly_figure(metric_name, ylabel, batch_size=None, restrict_model=None):
    fig = go.Figure()
    
    if batch_size is not None:
        # Single batch size plot
        for model in data:
            x = []
            y = []
            yerr = []
            for chunk_size in sorted(data[model]):
                values = [r[metric_name] for r in data[model][chunk_size][batch_size]]
                if values:
                    x.append(chunk_size)
                    y.append(np.mean(values))
                    yerr.append(np.std(values))
            if x:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    error_y=dict(type='data', array=yerr, visible=True),
                    mode='lines+markers',
                    name=f'{model}',
                    hovertemplate="Model: %{data.name}<br>" +
                                "Chunk Size: %{x}<br>" +
                                f"{ylabel}: %{{y}}<br>" +
                                
                                "<extra></extra>"
                ))
    else:
        # Batch size effect plot
        for model in data:
            if restrict_model is not None and restrict_model != model:
                continue
            batch_sizes = set(batch_size for chunk_size in data[model] for batch_size in data[model][chunk_size])
            
            for bs in sorted(batch_sizes):
                x = []
                y = []
                yerr = []
                for chunk_size in sorted(data[model]):
                    values = [r[metric_name] for r in data[model][chunk_size].get(bs, [])]
                    if values:
                        x.append(chunk_size)
                        y.append(np.mean(values))
                        yerr.append(np.std(values))
                if x:
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        error_y=dict(type='data', array=yerr, visible=True),
                        mode='lines+markers',
                        name=f'{model} (BS={bs})',
                        hovertemplate="Model: %{data.name}<br>" +
                                    "Chunk Size: %{x}<br>" +
                                    f"{ylabel}: %{{y}}<br>" +
                                    "<extra></extra>"
                    ))
    
    fig.update_layout(
        title=f'{ylabel} vs Chunk Size' + (f' (batch size of {batch_size})' if batch_size else ' (Batch Size Effect)'),
        xaxis_title='Chunk Size',
        yaxis_title=ylabel,
        hovermode='closest',
        showlegend=True
    )
    
    if metric_name == "minutes_per_sec":
        fig.update_layout(yaxis_range=[0, max_minutes_per_sec])
    
    return fig

def generate_html_report():
    html_content = """
    <html>
    <head>
        <title>RTF Analysis Results</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .plot-container {
                margin: 20px 0;
                padding: 10px;
                border: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
        <h1>RTF Analysis Results</h1>
    """
    
    # Generate plots for different batch sizes
    metrics = [
        ('rtf', 'Real-Time Factor (RTF)'),
        ('vram', 'Max VRAM Usage (MB)'),
        ('ram', 'Max CPU RAM Usage (MB)'),
        ('minutes_per_sec', 'Minutes of Audio Processed per Second')
    ]
    
    for metric, ylabel in metrics:
        # Single batch size plots
        for bs in [1, 2, 4, 8]:
            html_content += f'<div class="plot-container" id="plot_{metric}_bs{bs}"></div>'
            fig = create_plotly_figure(metric, ylabel, batch_size=bs)
            html_content += f"<script>{fig.to_json()}</script>"
            html_content += f"""
            <script>
                Plotly.newPlot('plot_{metric}_bs{bs}', {fig.to_json()});
            </script>
            """
        
        # Batch size effect plot
        html_content += f'<div class="plot-container" id="plot_{metric}_batch_effect"></div>'
        fig = create_plotly_figure(metric, ylabel)
        html_content += f"""
        <script>
            Plotly.newPlot('plot_{metric}_batch_effect', {fig.to_json()});
        </script>
        """
        
        # Individual model batch effect plots
        if metric in ['minutes_per_sec', 'vram']:
            for model in sorted(data):
                html_content += f'<div class="plot-container" id="plot_{metric}_{model}_batch_effect"></div>'
                fig = create_plotly_figure(metric, ylabel, restrict_model=model)
                html_content += f"""
                <script>
                    Plotly.newPlot('plot_{metric}_{model}_batch_effect', {fig.to_json()});
                </script>
                """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(f'{graph_dir}/summary.html', 'w') as f:
        f.write(html_content)

plot_metric('rtf', 'Real-Time Factor (RTF)')
plot_metric('vram', 'Max VRAM Usage (MB)')
plot_metric('ram', 'Max CPU RAM Usage (MB)')
plot_metric('minutes_per_sec', 'Minutes of Audio Processed per Second')

plot_metric('rtf', 'Real-Time Factor (RTF)', 2)
plot_metric('vram', 'Max VRAM Usage (MB)', 2)
plot_metric('ram', 'Max CPU RAM Usage (MB)', 2)
plot_metric('minutes_per_sec', 'Minutes of Audio Processed per Second', 2)

plot_metric('rtf', 'Real-Time Factor (RTF)', 4)
plot_metric('vram', 'Max VRAM Usage (MB)', 4)
plot_metric('ram', 'Max CPU RAM Usage (MB)', 4)
plot_metric('minutes_per_sec', 'Minutes of Audio Processed per Second', 4)

plot_metric('rtf', 'Real-Time Factor (RTF)', 8)
plot_metric('vram', 'Max VRAM Usage (MB)', 8)
plot_metric('ram', 'Max CPU RAM Usage (MB)', 8)
plot_metric('minutes_per_sec', 'Minutes of Audio Processed per Second', 8)

plot_batch_size_effect('rtf', 'Real-Time Factor (RTF)')
plot_batch_size_effect('vram', 'Max VRAM Usage (MB)')
plot_batch_size_effect('ram', 'Max CPU RAM Usage (MB)')
plot_batch_size_effect('minutes_per_sec', 'Minutes of Audio Processed per Second')

for model in sorted(data):
    plot_batch_size_effect('minutes_per_sec', 'Minutes of Audio Processed per Second', model)
    plot_batch_size_effect('vram', 'Max VRAM Usage (MB)', model)

# Add this at the end of your script
generate_html_report()

