import os, sys
import glob
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from extractor_models import KMerInteractionExtractor


class KMerStorage:
    """Parses 3, 4, and 5mers"""
    def __init__(self):
        self.pdbs = {k: dict() for k in [3, 4, 5]}
        self.three_mers = list()
        self.four_mers = list()
        self.five_mers = list()
        self._three_mer_tracker = 0
        self._four_mer_tracker = 0
        self._five_mer_tracker = 0

    def add_interaction(self, pdb, interactions):
        for i, interaction in enumerate(interactions):
            n = interaction.shape[0]
            if n != 0:
                if i == 0:
                    self.three_mers.append(interaction)
                    self.pdbs[3][pdb] = (self._three_mer_tracker, self._three_mer_tracker + n)
                    self._three_mer_tracker += n
                elif i == 1:
                    self.four_mers.append(interaction)
                    self.pdbs[4][pdb] = (self._four_mer_tracker, self._four_mer_tracker + n)
                    self._four_mer_tracker += n
                else:
                    self.five_mers.append(interaction)
                    self.pdbs[5][pdb] = (self._five_mer_tracker, self._five_mer_tracker + n)
                    self._five_mer_tracker += n
        return

    def compile_interactions(self):
        self.three_mers = torch.cat(self.three_mers) if len(self.three_mers) != 0 else torch.Tensor([])
        self.four_mers = torch.cat(self.four_mers) if len(self.four_mers) != 0 else torch.Tensor([])
        self.five_mers = torch.cat(self.five_mers) if len(self.five_mers) != 0 else torch.Tensor([])
        return

def parse_outputs(outdir, n, outname):
    count3, count4, count5 = 0, 0, 0
    d = {'pdbs': {i: dict() for i in range(3,6)}, 'interactions': {j: list() for j in range(3,6)}}
    for i in tqdm(range(n)):
        name = f'{outdir}/{i}.pkl'
        if os.path.isfile(name):
            with open(name, 'rb') as f:
                    tmp = pickle.load(f)

            d['pdbs'][3].update({name: (x+count3, y+count3) for name, (x, y) in tmp.pdbs[3].items()})
            d['pdbs'][4].update({name: (x+count4, y+count4) for name, (x, y) in tmp.pdbs[4].items()})
            d['pdbs'][5].update({name: (x+count5, y+count5) for name, (x, y) in tmp.pdbs[5].items()})
            threemer = tmp.three_mers
            fourmer = tmp.four_mers
            fivemer = tmp.five_mers
            d['interactions'][3].append(threemer)
            d['interactions'][4].append(fourmer)
            d['interactions'][5].append(fivemer)
            count3 += len(threemer)
            count4 += len(fourmer)
            count5 += len(fivemer)

    for k, v in d['interactions'].items():
        d['interactions'][k] = torch.cat(v).to(torch.int32)

    torch.save(d, outname)

def main():
    pdbdir = '/scratch/groups/possu/cath/train_pdb_store'
    outdir = 'kmer_interactions'
    n = int(sys.argv[1])
    t = 500

    # create kmer parser
    extractor = KMerInteractionExtractor()

    # get files
    files = glob.glob(f'{pdbdir}/*')
    split_files = np.array_split(files, t)[n]

    storage_unstructured = KMerStorage()
    for name in tqdm(split_files):
        interactions = [
            extractor.extract_interactions(name, 3, structured=False),
            extractor.extract_interactions(name, 4, structured=False), 
            extractor.extract_interactions(name, 5, structured=False)
        ]
        storage_unstructured.add_interaction(os.path.basename(name), interactions)
    
    storage_unstructured.compile_interactions()
    with open(f'{outdir}/unstructured/{n}.pkl', 'wb') as f:
        pickle.dump(storage_unstructured, f)
        
    storage_structured = KMerStorage()
    for name in tqdm(split_files):
        interactions = [
            extractor.extract_interactions(name, 3, structured=True),
            extractor.extract_interactions(name, 4, structured=True), 
            extractor.extract_interactions(name, 5, structured=True)
        ]
        storage_structured.add_interaction(os.path.basename(name), interactions)
    
    storage_structured.compile_interactions()
    with open(f'{outdir}/structured/{n}.pkl', 'wb') as f:
        pickle.dump(storage_structured, f)
    
if __name__ == "__main__":
    main()
