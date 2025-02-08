import os, sys, time
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from writer_models import KMerWriter

parser = argparse.ArgumentParser()
parser.add_argument('--interaction_file', type=str, default='structured_interactions.pt')
parser.add_argument('--pdb', type=str, default='mcl1/mcl1.pdb')
parser.add_argument('--chain', type=str, default='A')
parser.add_argument('--outdir', type=str, default='mcl1')
parser.add_argument('-s', '--s', type=int, required=True, help='Stage: either 0, 1, 2, or 3')
parser.add_argument('-n', '--n', type=int, default=0, help='Only applicable for stage 2')
parser.add_argument('-t', '--t', type=int, default=1, help='Only applicable for stage 2')
parser.add_argument('--n_samples_3', type=int, default=100)
parser.add_argument('--n_samples_4', type=int, default=100)
parser.add_argument('--n_samples_5', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# Add 1-indexed epitope residues here
residx = list(range(10, 21)) + list(range(36,65)) + list(range(96,110))
match_params = {3: 0.04, 4: 0.055, 5: 0.065}

# instantiate writer objects
writer = KMerWriter(args.interaction_file, match_params=match_params)
os.makedirs(args.outdir, exist_ok=True)
random.seed(args.seed)

if args.s == 0:
    match_params = [
        (0.03, 0.04, 0.05),
        (0.04, 0.05, 0.06),
        (0.05, 0.06, 0.07),
        (0.1, 0.15, 0.2),
        (0.15, 0.2, 0.25),
        (0.2, 0.25, 0.3)
    ]

    for setting in match_params:
        matches = writer.check_number_matches(args.pdb, args.chain, residx, *setting)
        print({kmer: (param, matches[kmer]) for param, kmer in zip(setting, matches.keys())})

elif args.s == 1:
    outnames = [(i, f'{args.outdir}/compiled_interactions_{i}.pt') for i in range(3,6)]
    for kmer, outname in tqdm(outnames):
        outname = f'{args.outdir}/compiled_interactions_{kmer}.pt'
        writer.compile_interactions(args.pdb, args.chain, residx, kmer, outname)
    
    d = dict()
    for _, outname in outnames:
        d.update(torch.load(outname))
        os.remove(outname)
    
    torch.save(d, f'{args.outdir}/compiled_interactions.pt')

elif args.s == 2:
    d = torch.load(f'{args.outdir}/compiled_interactions.pt')

    collected = dict()
    for kmer, direction in d.items():
        collected[kmer] = dict()
        for f_w, residue_wise in direction.items():
            collected[kmer][f_w] = dict()
            for idx, interaction in residue_wise.items():
                collected[kmer][f_w][idx] = np.array_split(interaction, args.t)[args.n]
    
    prepared_interactions = writer.prepare_interaction_lookup(collected)
    writer.parse_interaction_coordinates(args.pdb, args.chain, residx, prepared_interactions, f'{args.outdir}/fragments_{args.n}.pt')

    if args.n == args.t - 1:
        filenames = [f'{args.outdir}/fragments_{i}.pt' for i in range(args.t)]
        while not all(list(map(os.path.isfile, filenames))):
            print('Sleeping for 30s')
            time.sleep(30)
        
        results = dict()
        for name in filenames:
            for k, v in torch.load(name).items():
                if k not in results:
                    results[k] = list()
                results[k].append(v)
            os.remove(name)

        torch.save({k: torch.cat(v, dim=0) for k, v in results.items()}, f'{args.outdir}/fragments.pt')

elif args.s == 3:
    d = torch.load(f'{args.outdir}/fragments.pt')
    if 3 in d and args.n_samples_3 != 0:
        sample = random.sample(list(range(len(d[3]))), min(args.n_samples_3, len(d[3])))
        writer.write_pdb(d[3][sample], f'{args.outdir}/3mers.pdb')
    if 4 in d and args.n_samples_4 != 0:
        sample = random.sample(list(range(len(d[4]))), min(args.n_samples_4, len(d[4])))
        writer.write_pdb(d[4][sample], f'{args.outdir}/4mers.pdb')
    if 5 in d and args.n_samples_5 != 0:
        sample = random.sample(list(range(len(d[5]))), min(args.n_samples_5, len(d[5])))
        writer.write_pdb(d[5][sample], f'{args.outdir}/5mers.pdb')

else:
    raise ValueError('Invalid stage!')
