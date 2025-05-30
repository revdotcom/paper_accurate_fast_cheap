#!/usr/bin/env python3
import os, sys
import argparse
import glob
from collections import defaultdict
import json
import statistics
from multiprocessing import Pool

conversational_filler = [
    'UH', 'UHH', 'UM', 'EH', 'MM', 'HM', 'AH', 'HUH', 'HA', 'ER', 'OOF', 'HEE',
    'ACH', 'EEE', 'EW'
]
unk_tags = ['<UNK>', '<unk>']
gigaspeech_punctuations = [
    '<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>'
]
gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']
non_scoring_words = conversational_filler + unk_tags + \
    gigaspeech_punctuations + gigaspeech_garbage_utterance_tags


def clean_ref_files(ref_file_list, output_folder, suffix = "clean.txt" ):
    """Read all reference files and return dict of file_id -> text"""
    ref_dict = {}
    for ref_file in ref_file_list:
        file_id = os.path.splitext(os.path.basename(ref_file))[0]
        with open(ref_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            text = asr_text_post_processing(text)
            ref_dict[file_id] = text
        with open(os.path.join(output_folder, f"{file_id}.{suffix}"), 'w', encoding='utf-8') as f:
            f.write(text)   
    return ref_dict


def asr_text_post_processing(text):
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace('-', ' ')

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    return ' '.join(remaining_words)


def clean_ctm_files(ctm_file_list, output_folder, suffix = "clean.ctm"):
    """Read all CTM files and return dict of file_id -> combined text"""
    
    for ctm_file in ctm_file_list:
        file_id = os.path.splitext(os.path.basename(ctm_file))[0]
        print(f"Processing ctm {file_id}")
        cleaned_lines = []
        
        with open(ctm_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # Ensure line has enough fields
                    word = parts[4].upper()  # Convert to uppercase for comparison
                    if '-' in word:
                        wds = word.split('-')
                        for w in wds:
                            parts[4] = w
                            cleaned_lines.append(' '.join(parts).strip())
                    else:
                        if word not in non_scoring_words:
                            cleaned_lines.append(line.strip())
        
        # Write cleaned CTM file
        clean_ctm_path = os.path.join(output_folder, f"{file_id}.{suffix}")
        print(f"Writing cleaned ctm {clean_ctm_path}")
        with open(clean_ctm_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
    

def run_fstalign(cmd):
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script evaluates GigaSpeech ASR
                     result via fstalign tool''')
    parser.add_argument("--fstalign", type=str, default="fstalign", help="fstalign binary path")
    parser.add_argument("--accept-incomplete", action='store_true', help="will accept to work with missing ctm files")
    parser.add_argument(
        'ref',
        type=str,
        help="glob pattern for reference text files (*.ref_txt)")
    parser.add_argument(
        'hyp',
        type=str,
        help="glob pattern for CTM hypothesis files (*.ctm)")
    parser.add_argument(
        '--results',
        type=str,
        default="",
        help="results folder where to store all the intermediate files and final results")
    args = parser.parse_args()

    if args.results == "":
        args.results = os.path.join(os.path.dirname(args.hyp), "results")

    os.makedirs(args.results, exist_ok=True)

    # get the list of all ref files and all ctm files, compare them and get the list of files that are in ref but not in ctm
    ref_files = glob.glob(args.ref)
    ctm_files = glob.glob(args.hyp)
    
    # Create tuples of (basename, full_path) for both ref and ctm files
    ref_files_with_path = [(os.path.basename(file).split('.')[0], file) for file in ref_files]
    ctm_files_with_path = [(os.path.basename(file).split('.')[0], file) for file in ctm_files]

    print(f"len(ref_files_with_path): {len(ref_files_with_path)}")
    print(f"len(ctm_files_with_path): {len(ctm_files_with_path)}")
    
    # Compare basenames to find missing files
    ref_basenames = [basename for basename, _ in ref_files_with_path]
    ctm_basenames = [basename for basename, _ in ctm_files_with_path]
    files_in_ref_not_in_ctm = [basename for basename in ref_basenames if basename not in ctm_basenames]
    
    if len(files_in_ref_not_in_ctm) > 0:
        print(f"Warning: {len(files_in_ref_not_in_ctm)} files in ref but not in ctm:")
        print(files_in_ref_not_in_ctm)  
        if not args.accept_incomplete:
            sys.exit(1)
        # Filter out missing files from ref_files_with_path
        ref_files_with_path = [(basename, path) for basename, path in ref_files_with_path 
                              if basename not in files_in_ref_not_in_ctm]

    clean_ref_files([path for _, path in ref_files_with_path], args.results)
    clean_ctm_files([path for _, path in ctm_files_with_path], args.results)

    file_ids = [basename for basename, _ in ref_files_with_path]

    jsons_outputs = []


    fst_align_commands = []

    for file_id in file_ids:
        ref_file = os.path.join(args.results, f"{file_id}.clean.txt")
        ctm_file = os.path.join(args.results, f"{file_id}.clean.ctm")
        json_output = os.path.join(args.results, f"{file_id}.json")
        log_output = os.path.join(args.results, f"{file_id}.log")
        sbs_output = os.path.join(args.results, f"{file_id}.sbs")
        jsons_outputs.append(json_output)


        # check that  the ref and ctm files exist
        if not os.path.exists(ref_file):
            print(f"Error: {ref_file} does not exist")
            sys.exit(1)
        if not os.path.exists(ctm_file):
            print(f"Error: {ctm_file} does not exist")
            sys.exit(1)

        cmd = f"{args.fstalign} wer --ref '{ref_file}' --hyp '{ctm_file}' --json-log '{json_output}' --log '{log_output}' --output-sbs '{sbs_output}'"
        fst_align_commands.append(cmd)

    # execute all the fstalign commands in parallel, with a max of 10 processes, wait for all of them to finish
    with Pool(processes=10) as pool:
        pool.map(run_fstalign, fst_align_commands)
        pool.close()

    # wait for all the fstalign commands to finish

#     head tmp_gs_dev_A/rnnt_beam_search/POD1000000004.out.json -n 20
# {
# 	"wer" : 
# 	{
# 		"bestWER" : 
# 		{
# 			"deletions" : 113,
# 			"insertions" : 129,
# 			"meta" : {},
# 			"numErrors" : 558,
# 			"numWordsInReference" : 6807,
# 			"precision" : 0.93477940559387207,
# 			"recall" : 0.93697667121887207,
# 			"substitutions" : 316,
# 			"wer" : 0.081974439322948456
# 		},


    # read all the json files and get the micro and macro average wer, average deletions, insertions, substitutions
    total_deletions = 0
    total_insertions = 0
    total_substitutions = 0
    total_num_words_in_reference = 0
    total_num_errors = 0
    wer_values = []
    for json_output in jsons_outputs:
        with open(json_output, 'r', encoding='utf-8') as f:
            data = json.load(f)
            wer = data['wer']['bestWER']['wer']
            wer_values.append(wer * 100)  # Convert to percentage
            deletions = data['wer']['bestWER']['deletions']
            insertions = data['wer']['bestWER']['insertions']
            substitutions = data['wer']['bestWER']['substitutions']
            num_words_in_reference = data['wer']['bestWER']['numWordsInReference']
            num_errors = data['wer']['bestWER']['numErrors']

            total_deletions += deletions
            total_insertions += insertions
            total_substitutions += substitutions
            total_num_words_in_reference += num_words_in_reference
            total_num_errors += num_errors

    average_wer = total_num_errors / total_num_words_in_reference
    average_deletions = total_deletions / len(file_ids)
    average_insertions = total_insertions / len(file_ids)
    average_substitutions = total_substitutions / len(file_ids) 


    std_dev = statistics.stdev(wer_values) if len(wer_values) > 1 else 0

    # Calculate rates
    insertion_rate = (total_insertions / total_num_words_in_reference) * 100
    deletion_rate = (total_deletions / total_num_words_in_reference) * 100
    substitution_rate = (total_substitutions / total_num_words_in_reference) * 100
    total_wer = (total_num_errors / total_num_words_in_reference) * 100

    # Write the results in a summary.txt file in the results folder
    with open(os.path.join(args.results, "summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"TOTAL WER ({len(file_ids)}): {total_num_errors}/{total_num_words_in_reference} = {total_wer:.2f}%, Standard Deviation = {std_dev:.2f}%\n")
        f.write(f"Insertion Rate: {total_insertions}/{total_num_words_in_reference} = {insertion_rate:.2f}%\n")
        f.write(f"Deletion Rate: {total_deletions}/{total_num_words_in_reference} = {deletion_rate:.2f}%\n")
        f.write(f"Substitution Rate: {total_substitutions}/{total_num_words_in_reference} = {substitution_rate:.2f}%\n")

    








