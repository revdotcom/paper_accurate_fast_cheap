import sys
import glob
import numpy as np
import re
from collections import defaultdict
import os
from tqdm import tqdm

def parse_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    rtf_match = re.search(r'final_rtf (\d+\.\d+)', content)
    vram_match = re.search(r'max_vram=(\d+\.\d+)', content)
    ram_match = re.search(r'max_cpu_ram=(\d+\.\d+)', content)
    #minutes_match = re.search(r'(\d+\.\d+) minutes of audio processed per sec', content)
    minutes_match = re.search(r'final_rtf \d+\.\d+,\s+(\d+\.\d+) minutes of audio processed per sec', content, re.M)
    
    if rtf_match and vram_match and ram_match and minutes_match:
        return {
            'rtf': float(rtf_match.group(1)),
            'vram': float(vram_match.group(1)),
            'ram': float(ram_match.group(1)),
            'minutes_per_sec': float(minutes_match.group(1))
        }
    return None

def parse_config(file_path):
    parts = file_path.split('/')
    config = parts[-3]
    model, rest = config.split('-cs')
    chunk_size = int(rest.split('.')[0])
    batch_size = int(re.search(r'bs(\d+)', config).group(1))
    device = 'GPU' if 'gpu' in config else 'CPU'
    model = model + "-" + device
    return model, chunk_size, batch_size, device

def format_value(value, std_dev=None, metric=None):
    if std_dev is not None and abs(std_dev) < 1e-10:  # If std_dev is effectively zero
        std_dev = None
    
    if metric == 'rtf':
        # Use scientific notation with 2 significant digits for RTF
        if std_dev is None:
            return f"{value:.2e}"
        return f"{value:.2e} ±{std_dev:.2e}"
    else:
        # Keep regular formatting for other metrics
        if std_dev is None:
            return f"{value:.2f}"
        return f"{value:.2f} ±{std_dev:.2f}"

def format_cell(text, width):
    """Format cell content with fixed width, ensuring proper spacing"""
    return f" {text.center(width)} "

def get_column_widths(data, metric_name, models, chunk_sizes, batch_size=None):
    """Calculate the maximum width needed for each column"""
    widths = {'model': max(len(model) for model in models)}
    
    for cs in chunk_sizes:
        column_values = []
        for model in models:
            if cs in data[model]:
                if batch_size is not None:
                    values = [r[metric_name] for r in data[model][cs].get(batch_size, [])]
                else:
                    values = [r[metric_name] for bs in data[model][cs] for r in data[model][cs][bs]]
                
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    column_values.append(format_value(mean, std, metric_name))
                else:
                    column_values.append('-')
            else:
                column_values.append('-')
        
        cs_header = f"CS {cs}"
        widths[cs] = max(
            max(len(val) for val in column_values),
            len(cs_header)
        )
    
    return widths

def get_column_widths_flipped(data, metric_name, models, chunk_sizes, batch_size=None):
    """Calculate the maximum width needed for each column with models as columns"""
    # Ensure minimum width of 12 characters for readability
    widths = {'chunk_size': max(12, max(len(str(cs)) for cs in chunk_sizes))}
    
    for model in models:
        column_values = []
        for cs in chunk_sizes:
            if cs in data[model]:
                if batch_size is not None:
                    values = [r[metric_name] for r in data[model][cs].get(batch_size, [])]
                else:
                    values = [r[metric_name] for bs in data[model][cs] for r in data[model][cs][bs]]
                
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    column_values.append(format_value(mean, std, metric_name))
                else:
                    column_values.append('-')
            else:
                column_values.append('-')
        
        # Ensure minimum width of 12 characters
        widths[model] = max(12, max(len(val) for val in column_values), len(model))
    
    return widths

def generate_metric_table(data, metric_name, batch_size=None):
    # Collect all unique chunk sizes and models
    chunk_sizes = sorted(set(cs for model_data in data.values() for cs in model_data.keys()))
    models = sorted(data.keys())
    
    # Get column widths
    widths = get_column_widths_flipped(data, metric_name, models, chunk_sizes, batch_size)
    
    # Create table header
    table = f"### {metric_name} Data"
    if batch_size:
        table += f" (Batch Size {batch_size})\n\n"
    else:
        table += " (All Batch Sizes)\n\n"
    
    # Add model header row
    header = f"|{format_cell('Chunk Size', widths['chunk_size'])}|"
    header += "".join(f"{format_cell(model, widths[model])}|" for model in models)
    table += header + "\n"
    
    # Add separator row
    separator = f"|{'-' * (widths['chunk_size'] + 2)}|"
    separator += "".join(f"{'-' * (widths[model] + 2)}|" for model in models)
    table += separator + "\n"
    
    # Add data rows
    for cs in chunk_sizes:
        row = f"|{format_cell(str(cs), widths['chunk_size'])}|"
        for model in models:
            if cs in data[model]:
                if batch_size is not None:
                    values = [r[metric_name] for r in data[model][cs].get(batch_size, [])]
                else:
                    values = [r[metric_name] for bs in data[model][cs] for r in data[model][cs][bs]]
                
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    value = format_value(mean, std, metric_name)
                    row += f"{format_cell(value, widths[model])}|"
                else:
                    row += f"{format_cell('-', widths[model])}|"
            else:
                row += f"{format_cell('-', widths[model])}|"
        table += row + "\n"
    
    return table + "\n"

def generate_model_table(data, model, metric_name):
    """Generate a table for a specific model with chunk sizes as rows and batch sizes as columns"""
    # Collect all unique batch sizes and chunk sizes for this model
    batch_sizes = sorted(set(bs for cs_data in data[model].values() for bs in cs_data.keys()))
    chunk_sizes = sorted(data[model].keys())
    
    # Calculate column widths
    widths = {'chunk_size': max(12, max(len(str(cs)) for cs in chunk_sizes))}
    for bs in batch_sizes:
        column_values = []
        for cs in chunk_sizes:
            if cs in data[model] and bs in data[model][cs]:
                values = [r[metric_name] for r in data[model][cs][bs]]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    column_values.append(format_value(mean, std, metric_name))
                else:
                    column_values.append('-')
            else:
                column_values.append('-')
        
        bs_header = f"BS {bs}"
        widths[bs] = max(12, max(len(val) for val in column_values), len(bs_header))
    
    # Create table header
    table = f"### {model} - {metric_name}\n\n"
    
    # Add batch size header row
    header = f"|{format_cell('Chunk Size', widths['chunk_size'])}|"
    header += "".join(f"{format_cell(f'BS {bs}', widths[bs])}|" for bs in batch_sizes)
    table += header + "\n"
    
    # Add separator row
    separator = f"|{'-' * (widths['chunk_size'] + 2)}|"
    separator += "".join(f"{'-' * (widths[bs] + 2)}|" for bs in batch_sizes)
    table += separator + "\n"
    
    # Add data rows
    for cs in chunk_sizes:
        row = f"|{format_cell(str(cs), widths['chunk_size'])}|"
        for bs in batch_sizes:
            if cs in data[model] and bs in data[model][cs]:
                values = [r[metric_name] for r in data[model][cs][bs]]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    value = format_value(mean, std, metric_name)
                    row += f"{format_cell(value, widths[bs])}|"
                else:
                    row += f"{format_cell('-', widths[bs])}|"
            else:
                row += f"{format_cell('-', widths[bs])}|"
        table += row + "\n"
    
    return table + "\n"

def main():
    base_dir = "results.encoder-rtf"
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]

    output_dir = f"{base_dir}/tables"
    glob_pattern = f"{base_dir}/runs/*/rtf/*.rtf"
    os.makedirs(output_dir, exist_ok=True)
    files_to_process = glob.glob(glob_pattern)

    # Process data
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    valid_files = []
    batch_sizes = set()  # Keep track of all batch sizes
    
    for file in tqdm(list(files_to_process), desc="Processing files"):
        try:
            result = parse_file(file)
            if result:
                model, chunk_size, batch_size, device = parse_config(file)
                data[model][chunk_size][batch_size].append(result)
                valid_files.append(file)
                batch_sizes.add(batch_size)  # Add batch size to set
        except Exception as ex:
            print(f"Skipping {file}: {ex}")

    # Generate markdown content
    markdown_content = "# RTF Analysis Results\n\n"
    
    metrics = {
        #'rtf': 'Real-Time Factor (RTF)',
        'vram': 'Max VRAM Usage (MB)',
        #'ram': 'Max CPU RAM Usage (MB)',
        'minutes_per_sec': 'Minutes of Audio Processed per Second'
    }
    
    # First section: Tables grouped by metric and batch size
    markdown_content += "## Results by Metric and Batch Size\n\n"
    for metric, metric_name in metrics.items():
        markdown_content += f"### {metric_name}\n\n"
        for bs in sorted(batch_sizes):
            markdown_content += generate_metric_table(data, metric, bs)
    
    # Second section: Tables grouped by model
    markdown_content += "## Results by Model\n\n"
    for model in sorted(data.keys()):
        markdown_content += f"### {model}\n\n"
        for metric, metric_name in metrics.items():
            markdown_content += generate_model_table(data, model, metric)

    # Write to file
    output_file = f"{output_dir}/summary.md"
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"Generated markdown tables at: {output_file}")

if __name__ == "__main__":
    main() 
