import sys

# merge segmented transcriptions into a single transcription per file
# cat segments | pythonsegments_to_file.py /path/to/folder_to_store_txt 

prev_file = ''
outf = None
files = dict()
for line in sys.stdin:
    utt_id = line.split()[0]
    file_id = '_'.join(utt_id.split('_')[:-1])

    if not file_id in files:
        files[file_id] = ""

    files[file_id] += ' '.join(line.split()[1:]) + " "
    outf.write(' '.join(line.split()[1:]) + " ")
    prev_file = file_id

for file_id in files:
    outf = open(sys.argv[1] + "/" + file_id + ".txt", 'w')
    outf.write(files[file_id])
    outf.close()
        
            
