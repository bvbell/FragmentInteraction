import sys
import glob
import numpy as np
import torch
from tqdm import tqdm

from extractor_models import StrandExtractor

pdbdir = '/scratch/groups/possu/cath/train_pdb_store'
outdir = 'bstrand_interactions'
n = int(sys.argv[1])
t = 100

files = glob.glob(f'{pdbdir}/*')
split_files = np.array_split(files, t)[n]
extractor = StrandExtractor()

d = {'A': {'trans': list(), 'rot': list()}, 'P': {'trans': list(), 'rot': list()}}
for name in tqdm(split_files):
    for k1, v1 in extractor.extract_strand_transformations(name).items():
        for k2, v2 in v1.items():
            if len(v2) != 0:
                d[k1][k2].append(v2)

rewritten = {'A': {'trans': list(), 'rot': list()}, 'P': {'trans': list(), 'rot': list()}}
for k1, v1 in d.items():
    for k2, v2 in v1.items():
        rewritten[k1][k2] = torch.cat(v2)

torch.save(rewritten, f'{outdir}/{n}.pt')
