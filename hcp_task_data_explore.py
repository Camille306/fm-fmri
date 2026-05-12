import glob
import os
import pandas as pd
import torch

# 1. Globbing all files matching the pattern
folder = './data/task_shen268_graph_img_unadj'
pattern = os.path.join(folder, '*_*_0')
files = glob.glob(pattern)

# 2. Parsing the filenames into parts: id, tasktype, and rest
records = []
for fname in files:
    base = os.path.basename(fname)
    parts = base.split('_')
    if len(parts) >= 3:
        subject_id = parts[0]
        task_type = parts[1]
        suffix = '_'.join(parts[2:])
        records.append({'filename': fname, 'subject_id': subject_id, 'task_type': task_type, 'suffix': suffix})
    else:
        records.append({'filename': fname, 'subject_id': None, 'task_type': None, 'suffix': None})

df = pd.DataFrame(records)
print(df.head())

# Save the pandas dataframe to CSV
df.to_csv('hcp_task_info.csv', index=False)
print("Saved dataframe to hcp_task_info.csv")

# 3. Try loading the first file with PyTorch and inspecting data structures
if len(files) > 0:
    test_file = files[0]
    try:
        data = torch.load(test_file, map_location='cpu')
        print(f"Successfully loaded: {test_file}")
        print("Type of data:", type(data))
        if isinstance(data, dict):
            print("Keys in the loaded dict:", list(data.keys()))
            for k, v in data.items():
                print(f"Key: {k} | Type: {type(v)} | Shape/len(if applicable):", 
                      getattr(v, 'shape', len(v) if hasattr(v, '__len__') else 'N/A'))
        elif isinstance(data, (list, tuple)):
            print(f"Loaded a {type(data)} of length {len(data)}")
            for i, item in enumerate(data):
                print(f"Index: {i} | Type: {type(item)} | Shape/len(if applicable):", 
                      getattr(item, 'shape', len(item) if hasattr(item, '__len__') else 'N/A'))
        else:
            print("Loaded data:", data)
    except Exception as e:
        print(f"Could not load file {test_file} with torch.load: {e}")
else:
    print("No files found matching the given pattern.")

    # ALSO PROCESS THE HCP-RESTING-FC FOLDER

import re

# For ./data/hcp-resting-fc/{sub_id}/fc/REST{N}_LR_Shen268_corr_fc.npy
folder_rest = './data/hcp-resting-fc/'
records_rest_npys = []

# Glob for all REST*_LR_Shen268_corr_fc.npy files under all subject fc folders
pattern_rest_npy = os.path.join(folder_rest, '*', 'fc', 'REST*_LR_Shen268_corr_fc.npy')
files_rest_npy = glob.glob(pattern_rest_npy)

rest_regex = re.compile(r'REST(\d+)_LR_Shen268_corr_fc\.npy$')

for fpath in files_rest_npy:
    # fpath: .../{sub_id}/fc/REST{N}_LR_Shen268_corr_fc.npy
    try:
        # Extract subject id from path
        match = re.search(r'hcp-resting-fc/([^/]+)/fc/REST(\d+)_LR_Shen268_corr_fc\.npy$', fpath)
        if match:
            sub_id = match.group(1)
            rest_number = match.group(2)
        else:
            # fallback
            sub_id = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            fname = os.path.basename(fpath)
            rest_match = rest_regex.match(fname)
            rest_number = rest_match.group(1) if rest_match else None
        records_rest_npys.append({
            'filename': fpath,
            'subject_id': sub_id,
            'rest_number': rest_number
        })
    except Exception as exc:
        print(f"Error parsing: {fpath}. Exception: {exc}")
        records_rest_npys.append({
            'filename': fpath,
            'subject_id': None,
            'rest_number': None
        })

df_rest_npys = pd.DataFrame(records_rest_npys)
print(df_rest_npys.head())

# Save to CSV
df_rest_npys.to_csv('rest_info.csv', index=False)
print("Saved dataframe to rest_info.csv")

