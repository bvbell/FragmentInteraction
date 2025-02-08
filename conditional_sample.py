import os
import random
import pickle
import numpy as np
import torch
from typing import Union
from Bio.PDB import PDBParser
from tqdm import tqdm

import diffusion_sculpt_utils as utils
import common.pdb_data as pdb_data
from common.run_manager import RunManager
from sculptor.models.sculpt_utils_cp import write_ab_pdb


def kabsch(sample: torch.Tensor, template: torch.Tensor) -> tuple[torch.Tensor]:
    """Uses Kabsch algorithm to calculate optimal rotation and translation matrices"""
    sample_centroid = sample.mean(axis=0)
    template_centroid = template.mean(axis=0)
    T = template_centroid - sample_centroid

    centered_sample = sample - sample_centroid
    centered_template = template - template_centroid

    H = centered_sample.T @ centered_template
    U, S, Vt = torch.linalg.svd(H)
    if torch.linalg.det(Vt.T @ U.T) < 0.0:
        Vt[-1, :] *= -1.0

    R = Vt.T @ U.T
    return T, R

def calc_rmsd(peptide: torch.Tensor, prop_ab: torch.Tensor, residx: list) -> torch.Tensor:
    """Calculates L2 distance"""
    if len(peptide.shape) == 2: peptide = peptide.reshape(-1, 5, 3)
    if len(prop_ab.shape) == 3: prop_ab = prop_ab.reshape(-1, 3)

    motif = peptide[residx][:, [1,4]].reshape(-1,3)
    T, R = kabsch(motif, prop_ab)
    motif_centroid = motif.mean(dim=0)
    aligned = ((motif - motif_centroid) @ R.T) + motif_centroid + T
    return torch.sqrt((aligned - prop_ab).pow(2).sum() / aligned.shape[0])

def push_interactions_offcenter(peptide: torch.Tensor, prop_ab: torch.Tensor, residx: list) -> torch.Tensor:
    if len(peptide.shape) == 2: peptide = peptide.reshape(-1, 5, 3)
    if len(prop_ab.shape) == 3: prop_ab = prop_ab.reshape(-1, 3)

    motif = peptide[residx][:, [1, 4]].reshape(-1, 3)
    T, R = kabsch(prop_ab, motif)
    prop_ab_centroid = prop_ab.mean(dim=0)
    aligned = ((prop_ab - prop_ab_centroid) @ R.T) + prop_ab_centroid + T
    return aligned.reshape(-1, 6)

def sample_optimal_alignment_single_fragment(length, z, config, model, ode_noise, interaction, target_coord, clash_distance_cutoff, match_distance_cutoff):
    param_tracker = list()
    kmer_length = interaction.shape[0]
    starting_peptide = utils.binder_sample(length, z, config, model, ode_noise=ode_noise)
    has_clashes = lambda peptide_coord: torch.any(torch.norm(peptide_coord[:, None] - target_coord[None], dim=-1) < clash_distance_cutoff)
    for i in range(1, length-kmer_length+1):
        interaction_idx = list(range(i, i+kmer_length))
        prop_ab = interaction[:, [1,4]].reshape(-1,3)
        offset_prop_ab = push_interactions_offcenter(starting_peptide, prop_ab, interaction_idx)
        loss_fn = lambda x0, _: utils.l2_loss(x0, length, interaction_idx, offset_prop_ab, method='distance_matrix')
        x0 = utils.binder_sample(length, z, config, model, ode_noise=ode_noise, do_replacement=True, template=starting_peptide, prop_ab_coord=offset_prop_ab, interface=interaction_idx, align_origin=False, loss_fn=loss_fn)

        # process output
        peptide_motif = x0.reshape(-1, 5, 3)[interaction_idx][:, [1,4]].reshape(-1, 3)
        motif_centroid = peptide_motif.mean(dim=0)
        T, R = kabsch(peptide_motif, prop_ab)
        aligned_peptide = ((x0 - motif_centroid) @ R.T) + motif_centroid + T
        distance = calc_rmsd(x0, prop_ab, interaction_idx).item()
        if not has_clashes(aligned_peptide):
            param_tracker.append((z, ode_noise, interaction, interaction_idx, x0, aligned_peptide, distance))
    
    return [param for param in param_tracker if param[-1] < match_distance_cutoff]

def sample_optimal_alignment_two_fragment(length, z, config, model, ode_noise, interaction, residx, target_coord, clash_distance_cutoff, match_distance_cutoff):
    param_tracker = list()
    starting_peptide = utils.binder_sample(length, z, config, model, ode_noise=ode_noise)
    has_clashes = lambda peptide_coord: torch.any(torch.norm(peptide_coord[:, None] - target_coord[None], dim=-1) < clash_distance_cutoff)
    interaction_length = max(residx) - min(residx)
    for i in range(length - interaction_length):
        interaction_idx = [x - residx[0] + i for x in residx]
        prop_ab = interaction[:, [1,4]].reshape(-1,3)
        offset_prop_ab = push_interactions_offcenter(starting_peptide, prop_ab, interaction_idx)
        loss_fn = lambda x0, _: utils.l2_loss(x0, length, interaction_idx, offset_prop_ab, method='distance_matrix')
        x0 = utils.binder_sample(length, z, config, model, ode_noise=ode_noise, do_replacement=True, template=starting_peptide, prop_ab_coord=offset_prop_ab, interface=interaction_idx, align_origin=False, loss_fn=loss_fn)

        # process output
        peptide_motif = x0.reshape(-1, 5, 3)[interaction_idx][:, [1,4]].reshape(-1, 3)
        motif_centroid = peptide_motif.mean(dim=0)
        T, R = kabsch(peptide_motif, prop_ab)
        aligned_peptide = ((x0 - motif_centroid) @ R.T) + motif_centroid + T
        distance = calc_rmsd(x0, prop_ab, interaction_idx).item()
        if not has_clashes(aligned_peptide):
            param_tracker.append((z, ode_noise, interaction, interaction_idx, x0, aligned_peptide, distance))
    
    return [param for param in param_tracker if param[-1] < match_distance_cutoff]

def sample_single_fragment(model, config, interactions, target_coord, clash_distance_cutoff=2.5, match_distance_cutoff=1.5, n_iter=1):
    device = model.device
    kmer_length = random.choice(list(interactions.keys()))
    possible_interactions = interactions[kmer_length]
    interaction_idx = random.choice(range(len(possible_interactions)))
    chosen_interaction = interactions[kmer_length][interaction_idx]
    results = list()
    for _ in range(n_iter):
        length = random.randint(11,13)
        z = utils.initialize_random_coords(length+1, config, device)
        ode_noise = torch.randn(200, 1, config.data.fixed_size, 5, 3).to(device)
        results += sample_optimal_alignment_single_fragment(length, z, config, model, ode_noise, chosen_interaction, target_coord, clash_distance_cutoff, match_distance_cutoff)
    return results

def sample_multiple_fragments(model, config, target_coord, original_peptide, z, ode_noise, interactions, residx, clash_distance_cutoff=2.5, match_distance_cutoff=1.5, n_iter=1):
    results = list()
    length = (original_peptide.shape[0] // 5) - 1
    prop_ab = interactions[:, [1,4]].reshape(-1,3)
    offset_prop_ab = push_interactions_offcenter(original_peptide, prop_ab, residx)
    loss_fn = lambda x0, _: utils.l2_loss(x0, length, residx, offset_prop_ab, method='distance_matrix')
    x0 = utils.binder_sample(length, z, config, model, ode_noise=ode_noise, do_replacement=True, template=original_peptide, prop_ab_coord=offset_prop_ab, interface=residx, align_origin=False, loss_fn=loss_fn)

    # process output
    peptide_motif = x0.reshape(-1, 5, 3)[residx][:, [1,4]].reshape(-1, 3)
    motif_centroid = peptide_motif.mean(dim=0)
    T, R = kabsch(peptide_motif, prop_ab)
    aligned_peptide = ((x0 - motif_centroid) @ R.T) + motif_centroid + T
    distance = calc_rmsd(x0, prop_ab, residx).item()
    has_clashes = torch.any(torch.norm(aligned_peptide[:, None] - target_coord[None], dim=-1) < clash_distance_cutoff).item()
    if not has_clashes and distance < match_distance_cutoff:
        results.append((z, ode_noise, interactions, residx, x0, aligned_peptide, distance))

    device = model.device
    for _ in range(n_iter):
        length = random.randint(11,13)
        z = utils.initialize_random_coords(length+1, config, device)
        ode_noise = torch.randn(200, 1, config.data.fixed_size, 5, 3).to(device)
        results += sample_optimal_alignment_two_fragment(length, z, config, model, ode_noise, interactions, residx, target_coord, clash_distance_cutoff, match_distance_cutoff)
    
    return results

def extract_target_coord(pdb_filename: str, chain: str) -> torch.Tensor:
    target = PDBParser(QUIET=True).get_structure('target', pdb_filename)[0][chain]
    coord = np.asarray([atom.coord for residue in target for atom in residue])
    return torch.from_numpy(coord)

def find_additional_matches(peptide_coord: torch.Tensor, interaction_d: dict[int,torch.Tensor], occupied_residues: list, distance_cutoff: float=5.0):
    peptide_length = peptide_coord.shape[0] // 5
    available_residues = [i for i in range(peptide_length) if i != 0 and i != peptide_length-1 and i not in occupied_residues]
    peptide_ca = peptide_coord.reshape(-1, 5, 3)[available_residues, 1]
    matches = dict()
    for kmer_length, interactions in interaction_d.items():
        interaction_ca = interactions[:, :, 1]
        dm = torch.norm(peptide_ca[:, None] - interaction_ca[:, None], dim=-1).mean(dim=-1)

        for interaction_idx, residx in torch.argwhere(dm < distance_cutoff).tolist():
            if kmer_length not in matches: matches[kmer_length] = dict()
            if interaction_idx not in matches[kmer_length]: matches[kmer_length][interaction_idx] = list()
            matches[kmer_length][interaction_idx].append(residx)
    
    # choose valid interactions
    valid_interactions = dict()
    for length, subdict in matches.items():
        for interaction, residues in subdict.items():
            valid_residues = list()
            if len(residues) < length:
                continue

            for i in range(len(residues)-length+1):
                if np.all(np.diff(residues[i:i+length]) == 1):
                    corrected_residx = [available_residues[j] for j in residues[i:i+length]]
                    valid_residues.append(corrected_residx)

            if len(valid_residues) != 0:
                if length not in valid_interactions: 
                    valid_interactions[length] = dict()
                valid_interactions[length][interaction] = valid_residues

    return valid_interactions

def compile_fragments(residues1, residues2, interaction1, interaction2):
    if residues1[0] < residues2[0]:
        all_residues = residues1 + residues2
        all_interactions = torch.cat([interaction1, interaction2], dim=0)
    else:
        all_residues = residues2 + residues1
        all_interactions = torch.cat([interaction2, interaction1], dim=0)
    return all_residues, all_interactions

def main():
    outdir = 'conditional_generation_check'
    interaction_file = 'cldn6_interactions/fragments.pt'
    molecule = 'thioether'
    manager = RunManager()
    manager.parse_args()
    args = manager.args
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config, mode_dicts = utils.initialize_model(args.config, args.model_chkpt, device, return_mode_dict=True)

    n = args.epoch
    random.seed(n)
    
    # parse data file
    interactions = {k: v.to(device) for k, v in torch.load(interaction_file).items()}
    target_coord = extract_target_coord('cldn6_interactions/cldn6_relaxed.pdb', 'A').to(device)
    
    results = list()
    for i in tqdm(range(5)):
        results += sample_single_fragment(model, config, interactions, target_coord, clash_distance_cutoff=2.5, match_distance_cutoff=1.0, n_iter=5)
        print('single', len(results))

    with open(f'cldn6_results/single_{n}.pkl', 'wb') as f:
        pickle.dump(results, f)

    multi_fragment_params = list()
    for z, ode_noise, interaction, residx, original_peptide, aligned_peptide, _ in results:
        additional_fragments = find_additional_matches(aligned_peptide, interactions, residx)
        if len(additional_fragments) == 0:
            continue

        for kmer_length, matched_interactions in additional_fragments.items():
            for matched_interaction_idx, new_residx_lst in matched_interactions.items():
                for new_residx in new_residx_lst:
                    all_residues, all_interactions = compile_fragments(residx, new_residx, interaction, interactions[kmer_length][matched_interaction_idx])
                    multi_fragment_params.append((original_peptide, z, ode_noise, all_interactions, all_residues))

    multi_results = list()
    for frag_params in multi_fragment_params:
        multi_results += sample_multiple_fragments(model, config, target_coord, *frag_params, n_iter=5)
        print('multi', len(multi_results))

    with open(f'cldn6_results/multi_{n}.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
