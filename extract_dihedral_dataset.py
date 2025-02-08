import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from extractor_models import DihedralDatabaseExtractor


def parse_outputs(outdir, n, outname):
    results = {i: {'forward': list(), 'reverse': list()} for i in range(3, 6)}
    for i in tqdm(range(n)):
        filename = f'{outdir}/{i}.pt'
        d = torch.load(filename)
        for k1, v1 in d.items():
            for k2, v2 in v1.items():
                results[k1][k2].append(v2)
    
    results = {k1: {k2: torch.cat(v2) for k2, v2 in v1.items()} for k1, v1 in results.items()}
    torch.save(results, outname)

def main():
    n, interaction_file, outname_dir = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    t = 100

    d = torch.load(interaction_file)
    results = {i: {'forward': list(), 'reverse': list()} for i in range(3,6)}
    extractor = DihedralDatabaseExtractor('/scratch/groups/possu/cath/pdb_store')
    
    # 3mers
    k3 = np.array_split(list(d['pdbs'][3].keys()), t)[n]
    interactions = d['interactions'][3].sort(dim=-1)[0]
    for name in tqdm(k3):
        begin, end = d['pdbs'][3][name]
        residx1, residx2 = interactions[begin:end, 0], interactions[begin:end, 1]
        dihedrals = extractor(name, 3, residx1, residx2)
        results[3]['forward'].append(dihedrals[:, 0])
        results[3]['reverse'].append(dihedrals[:, 1])

    # 4mers
    k4 = np.array_split(list(d['pdbs'][4]), t)[n]
    interactions = d['interactions'][4].sort(dim=-1)[0]
    for name in tqdm(k4):
        begin, end = d['pdbs'][4][name]
        residx1, residx2 = interactions[begin:end, 0], interactions[begin:end, 1]
        dihedrals = extractor(name, 4, residx1, residx2)
        results[4]['forward'].append(dihedrals[:, 0])
        results[4]['reverse'].append(dihedrals[:, 1])

    # 5mer
    k5 = np.array_split(list(d['pdbs'][5]), t)[n]
    interactions = d['interactions'][5].sort(dim=-1)[0]
    for name in tqdm(k5):
        begin, end = d['pdbs'][5][name]
        residx1, residx2 = interactions[begin:end, 0], interactions[begin:end, 1]
        dihedrals = extractor(name, 5, residx1, residx2)
        results[5]['forward'].append(dihedrals[:, 0])
        results[5]['reverse'].append(dihedrals[:, 1])

    results = {k1: {k2: torch.cat(v2) for k2, v2 in v1.items()} for k1, v1 in results.items()}
    torch.save(results, f'{outname_dir}/{n}.pt')

if __name__ == "__main__":
    main()
