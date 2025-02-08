import os
import numpy as np
import torch
from typing import Union
from Bio.PDB import PDBParser

from utils import StructUtils


class StrandExtractor(StructUtils):
    """Records the translation and rotation matrices between interaction beta strand H bonds"""
    def __init__(self):
        super().__init__()
        
    def _identify_strand_directions(self, coord: torch.Tensor, bstrand_stretches: list) -> list[str]:
        results = list()
        for stretch in bstrand_stretches:
            # collect residx that define the strand directions
            if len(stretch) == 1:
                strand1_residues = sorted([stretch[0][0], stretch[-1][0]])
                strand2_residues = sorted([stretch[0][1], stretch[-1][1]])
            else:                        
                strand1_residues = sorted([stretch[0][0], stretch[0][0] + 2 if (stretch[0][0] + 2) < coord.shape[0] else (stretch[0][0] - 2)])
                strand2_residues = sorted([stretch[0][1], stretch[0][1] + 2 if (stretch[0][1] + 2) < coord.shape[0] else (stretch[0][1] - 2)])
            # define vectors of each strand, res1 -> res2
            strand1_vec = coord[strand1_residues[1], 1] - coord[strand1_residues[0], 1]
            strand2_vec = coord[strand2_residues[1], 1] - coord[strand2_residues[0], 1]
            results.append(self.calc_angle(strand1_vec, strand2_vec))
            
        # P for parallel, A for antiparallel
        return ['P' if angle < torch.pi / 2 else 'A' for angle in results]
    
    def _extract_strands_helper(self, coord: torch.Tensor) -> tuple[dict, list]:
        strand_data = self.find_beta_strand_pairs(coord)
        if strand_data is None:
            return None
        
        _, all_pairs = strand_data
        bstrand_stretches = self._distinguish_strand_regions(all_pairs)
        bstrand_directions = self._identify_strand_directions(coord, bstrand_stretches)
        complete_strands = list()
        interactions = {'A': list(), 'P': list()}
        for strand, direction in zip(bstrand_stretches, bstrand_directions):
            if len(strand) != 1:
                strand1 = sorted([strand[0][0], strand[-1][0]])
                strand2 = sorted([strand[0][1], strand[-1][1]])
                complete_strands.append(('-'.join(map(str,strand1)), '-'.join(map(str,strand2)), direction))
            else:
                complete_strands.append(tuple([str(i) for i in sorted(strand[0])] + [direction]))
            
            interactions[direction] += strand
        return interactions, complete_strands
        
    def find_beta_strand_pairs(self, coord: torch.Tensor) -> Union[None, tuple[torch.Tensor, torch.Tensor]]:
        # Find beta strand pairs that will be analyzed
        onehot, hbmap = self.assign(coord)
        possible_strand_idx = torch.argwhere(onehot[:, -1]).flatten()
        if len(possible_strand_idx) == 0:
            return None
        
        # identify beta strand pairs
        idx_map = {i: elem for i, elem in enumerate(possible_strand_idx)}
        bstrand_hbmap = hbmap[possible_strand_idx][:, possible_strand_idx]
        valid_hbonds = torch.argwhere(bstrand_hbmap >= 0.5)
        
        remapped_pairs = torch.zeros_like(valid_hbonds)
        for i in range(len(valid_hbonds)):
            remapped_pairs[i, 0] = idx_map[valid_hbonds[i, 0].item()]
            remapped_pairs[i, 1] = idx_map[valid_hbonds[i, 1].item()]

        return possible_strand_idx, remapped_pairs
    
    def _distinguish_strand_regions(self, bstrand_pairs: torch.Tensor) -> list[tuple]:
        # remove symmetry from pairs
        to_remove = list()
        for i in range(len(bstrand_pairs)-1):
            reverse = torch.Tensor([bstrand_pairs[i, 1], bstrand_pairs[i, 0]])
            duplicate_entry = torch.argwhere(bstrand_pairs[i+1:] == reverse)
            if len(duplicate_entry) != 0:
                to_remove += [i+j+1 for j in torch.unique(duplicate_entry[:, 0]).tolist()]
        to_remove_set = set(to_remove)
        remaining_pairs = bstrand_pairs[[i for i in range(len(bstrand_pairs)) if i not in to_remove_set]]
        
        # collect indices of consecutive strands
        i = 0
        delta = torch.diff(remaining_pairs[:, 0]).tolist()
        double_tracker, single_tracker = False, False
        lone_step_counter, double_step_counter, single_step_counter = 0, 0, 0
        lone_step_regions, double_step_regions, single_step_regions = {0: list()}, {0: list()}, {0: list()}
        while i < len(delta):
            if delta[i] == 2:
                double_tracker = True
                if single_tracker:
                    single_step_regions[single_step_counter].append(tuple(remaining_pairs[i].tolist()))
                    double_step_counter += 1
                    single_tracker = False
                    double_step_regions[double_step_counter] = list()
                else:
                    double_step_regions[double_step_counter].append(tuple(remaining_pairs[i].tolist()))

            elif delta[i] == 1:
                single_tracker = True
                if double_tracker: 
                    double_step_regions[double_step_counter].append(tuple(remaining_pairs[i].tolist()))
                    single_step_counter += 1
                    double_tracker = False
                    single_step_regions[single_step_counter] = list()
                else:
                    single_step_regions[single_step_counter].append(tuple(remaining_pairs[i].tolist()))
            
            elif single_tracker:
                single_step_regions[single_step_counter].append(tuple(remaining_pairs[i].tolist()))
                single_step_counter += 1
                single_step_regions[single_step_counter] = list()
                single_tracker = False
            
            elif double_tracker:
                double_step_regions[double_step_counter].append(tuple(remaining_pairs[i].tolist()))
                double_step_counter += 1
                double_step_regions[double_step_counter] = list()
                double_tracker = False
            
            else:
                lone_step_regions[lone_step_counter] = [tuple(remaining_pairs[i].tolist())]
                lone_step_counter += 1

            if i == len(delta) - 1:
                if delta[i] == 2:
                    double_step_regions[double_step_counter].append(tuple(remaining_pairs[i+1].tolist()))
                elif delta[i] == 1:
                    single_step_regions[single_step_counter].append(tuple(remaining_pairs[i+1].tolist()))
                else:
                    lone_step_regions[lone_step_counter] = [tuple(remaining_pairs[i+1].tolist())]

            i += 1

        results = sorted(
            [v1 for v1 in lone_step_regions.values() if len(v1) != 0] +\
            [v2 for v2 in single_step_regions.values() if len(v2) != 0] +\
            [v3 for v3 in double_step_regions.values() if len(v3) != 0],
            key=lambda x: x[0]
        )

        if len(remaining_pairs) == 1 and len(results) == 0:
            results.append([tuple(remaining_pairs.flatten().tolist())])

        assert len([x for y in results for x in y]) == len(remaining_pairs)

        return results
    
    def _calculate_rot_trans(self, template: torch.Tensor, partner: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates optimal rotation and translation matrices using Kabsh"""
        if template.shape != partner.shape:
            raise ValueError('Two coordinate matrices do not have the same shape')
        elif len(template.shape) == 2:
            template = template[None]
            partner = partner[None]
            
        # get translation matrices
        template_centroid = template.mean(dim=1, keepdims=True)
        partner_centroid = partner.mean(dim=1, keepdims=True)
        opt_trans = template_centroid - partner_centroid
        
        # center points
        template_centered = template - template_centroid
        partner_centered = partner - partner_centroid
        
        # calculate covariance matrix with SVD
        H = partner_centered.transpose(2, 1) @ template_centered
        U, S, Vt = torch.linalg.svd(H)
        determinant = torch.linalg.det(Vt.transpose(2, 1) @ U.transpose(2, 1))
        Vt[determinant < 0.0, -1, :] *= -1.0
        opt_rot = Vt.transpose(2, 1) @ U.transpose(2, 1)

        return opt_trans[:, 0], opt_rot
    
    def _check_alignment(self, template: torch.Tensor, partner: torch.Tensor, trans: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Helper function to check alignment during debugging. Only accepts batched matrices. Returns rmsd."""
        partner_centroid = partner.mean(dim=1, keepdims=True)
        aligned = (partner - partner_centroid) @ angles.transpose(2, 1) + partner_centroid + trans
        return torch.sqrt((template - aligned).pow(2).sum(dim=-1).mean(dim=-1))
    
    def _correct_direction(self, results_d: dict) -> dict[str, torch.Tensor]:
        """Ensures that all vectors and matrices are represented in both directions"""
        cleaned_results = {'A': dict(), 'P': dict()}
        
        # transA
        if len(results_d['A']['trans']) != 0:
            transA = torch.cat(results_d['A']['trans'])
            cleaned_results['A']['trans'] = torch.cat([transA, -transA])
        else:
            cleaned_results['A']['trans'] = torch.Tensor([])
            
        # transP
        if len(results_d['P']['trans']) != 0:
            transP = torch.cat(results_d['P']['trans'])
            cleaned_results['P']['trans'] = torch.cat([transP, -transP])
        else:
            cleaned_results['P']['trans'] = torch.Tensor([])
            
        # rotA
        if len(results_d['A']['rot']) != 0:
            rotA = torch.cat(results_d['A']['rot'])
            doubled_rotA = torch.cat([rotA, rotA.transpose(2, 1)])
            cleaned_results['A']['rot'] = doubled_rotA
        else:
            cleaned_results['A']['rot'] = torch.Tensor([])
            
        # rotP
        if len(results_d['P']['rot']) != 0:
            rotP = torch.cat(results_d['P']['rot'])
            doubled_rotP = torch.cat([rotP, rotP.transpose(2, 1)])
            cleaned_results['P']['rot'] = doubled_rotP
        else:
            cleaned_results['P']['rot'] = torch.Tensor([])
            
        return cleaned_results
    
    def extract_strand_transformations(self, filename: str, check_alignment: bool=False) -> dict[str, torch.Tensor]:
        """Extracts all beta strand interactions from a pdb"""
        results = {'A': {'trans': list(), 'rot': list()}, 'P': {'trans': list(), 'rot': list()}}
        all_coords = torch.cat(self.parse_all_chains(filename))
        strand_data = self._extract_strands_helper(all_coords)
        if strand_data is None:
            return results
        interactions, _ = strand_data

        # first do antiparallel
        if len(interactions['A']) != 0:
            strand1_A, strand2_A = zip(*interactions['A'])
            template_coord_A = all_coords[list(strand1_A)]
            partner_coord_A = all_coords[list(strand2_A)]
            transA, rotA = self._calculate_rot_trans(template_coord_A, partner_coord_A)
            results['A']['trans'].append(transA)
            results['A']['rot'].append(rotA)
        else:
            template_coord_A = None
            partner_coord_A = None

        # then do parallel
        if len(interactions['P']) != 0:
            strand1_P, strand2_P = zip(*interactions['P'])
            template_coord_P = all_coords[list(strand1_P)]
            partner_coord_P = all_coords[list(strand2_P)]
            transP, rotP = self._calculate_rot_trans(template_coord_P, partner_coord_P)
            results['P']['trans'].append(transP)
            results['P']['rot'].append(rotP)
        else:
            template_coord_P = None
            partner_coord_P = None

        # for debugging
        if check_alignment:
            if template_coord_A is not None and partner_coord_A is not None:
                rmsd_A = self._check_alignment(template_coord_A, partner_coord_A, transA, rotA)
                rmsd_A_reversed = self._check_alignment(partner_coord_A, template_coord_A, -transA, rotA.transpose(2, 1))
                print(f'Antiparallel: {rmsd_A}\n')
                print(f'Antiparallel reversed direction: {rmsd_A_reversed}\n')
            
            if template_coord_P is not None and partner_coord_P is not None:
                rmsd_P = self._check_alignment(template_coord_P, partner_coord_P, transP, rotP)
                rmsd_P_reversed = self._check_alignment(partner_coord_P, template_coord_P, -transP, rotP.transpose(2, 1))
                print(f'Parallel: {rmsd_P}\n')
                print(f'Parallel reversed direction: {rmsd_P_reversed}\n')
            
        return self._correct_direction(results)

class KMerInteractionExtractor(StrandExtractor):
    """Extracts and optionally writes interacting structured motifs"""
    def __init__(self, cutoff_distance: float=7.5, residue_separation: int=10):
        super().__init__()
        self.cutoff = cutoff_distance
        self.buffer = residue_separation
        
    def _get_structured_indices(self, coord: torch.Tensor, kmer_length: int) -> list:
        """Finds the indices of all structured motifs"""
        secondary_structs = self.assign(coord)[0][:, 1:]
        motifs = torch.argwhere(secondary_structs[:, 0] | secondary_structs[:, 1]).flatten()
        if len(motifs) == 0:
            return list()
        
        motifs.sort()
        stretches = dict()
        tracker, previous = 0, motifs[0]
        for i, residx in enumerate(motifs[1:]):
            if i != 0:
                if residx - 1 == previous:
                    stretches[tracker].append(residx)
                    previous = residx
                else:
                    tracker += 1
                    stretches[tracker] = [residx]
                    previous = residx
            elif residx - 1 == previous:
                stretches[tracker] = [previous, residx]
                previous = residx
            else:
                stretches[tracker] = [previous]
                tracker += 1
                stretches[tracker] = [residx]
                previous = residx
        
        accepted_stretches = [torch.Tensor(stretch) for stretch in stretches.values() if len(stretch) >= kmer_length]
        return accepted_stretches

    def _extract_valid_kmers(self, kmer_length: int, interactions: torch.Tensor) -> torch.Tensor:
        """Extract stretches of interacting kmers"""
        if len(interactions) < kmer_length or len(torch.unique(interactions[:, 0])) < kmer_length or len(torch.unique(interactions[:, 1])) < 3:
            return None
        
        # extract continuous kmers
        residues1, _ = torch.sort(torch.unique(interactions[:, 0]))
        residues2, _ = torch.sort(torch.unique(interactions[:, 1]))
        stacked_res1 = [residues1[i:i+kmer_length] for i in range(len(residues1)-kmer_length+1)]
        stacked_res2 = [residues2[i:i+kmer_length] for i in range(len(residues2)-kmer_length+1)]
        if len(stacked_res1) == 0 or len(stacked_res2) == 0:
            return None
        possible_res1_kmers = torch.stack(stacked_res1)
        possible_res2_kmers = torch.stack(stacked_res2)
        res1_kmers = possible_res1_kmers[torch.all(torch.diff(possible_res1_kmers) == 1, dim=-1)]
        res2_kmers = possible_res2_kmers[torch.all(torch.diff(possible_res2_kmers) == 1, dim=-1)]
        # concatenate reversed kmers
        res2_kmers = torch.cat([res2_kmers, res2_kmers.flip(1)])
        
        # create all combinations
        construct_required_pairs = lambda kmer1, kmer2: [(x.item(), y.item()) for x, y in torch.stack([kmer1, kmer2]).T]
        possible_combos = [construct_required_pairs(row1, row2) for row1 in res1_kmers for row2 in res2_kmers]
        
        # check if necessary interactions exist
        interaction_pairs = set([(x.item(), y.item()) for x, y in interactions])
        all_pairs_present = lambda possible_pairs: all([pair in interaction_pairs for pair in possible_pairs])
        valid_kmers = [combo for combo in possible_combos if all_pairs_present(combo)]
        if len(valid_kmers) == 0:
            return None

        # convert valid pairs to a usable array
        def construct_matrix(combo):
            res1, res2 = zip(*combo)
            return torch.Tensor([list(res1), list(res2)])
        
        results = torch.stack([construct_matrix(combo) for combo in valid_kmers])
        return results
        
    def _find_interacting_motifs(self, coord: torch.Tensor, kmer_length: int, structured: bool=False) -> list:
        """Finds the indices of all structured, interacting motifs"""
        all_interactions = list()

        if structured:
            stretches = self._get_structured_indices(coord, kmer_length)
            if len(stretches) == 0:
                return torch.Tensor([])

            for i, j in zip(*np.triu_indices(len(stretches), k=1)):
                stretch1, stretch2 = stretches[i].to(int), stretches[j].to(int)
                c1 = coord[stretch1].reshape(-1, 3)
                c2 = coord[stretch2].reshape(-1, 3)
                dm = torch.linalg.norm(c1[:, None] - c2[None], dim=-1)
                close_res = torch.floor(torch.argwhere(dm < self.cutoff) / 4)
                unique_interactions = torch.unique(close_res, dim=0).to(int)
                if len(unique_interactions) != 0:
                    interaction_pairs = torch.stack([stretch1[unique_interactions[:, 0]], stretch2[unique_interactions[:, 1]]], dim=1)
                    valid_kmers = self._extract_valid_kmers(kmer_length, interaction_pairs)
                    if valid_kmers is not None:
                        all_interactions.append(valid_kmers)

        else:
            stretch = torch.arange(1, len(coord)-1)
            c1 = coord[stretch].reshape(-1, 3)
            c2 = coord[stretch].reshape(-1, 3)
            dm = torch.linalg.norm(c1[:, None] - c2[None], dim=-1)
            close_res = torch.floor(torch.argwhere(dm < self.cutoff) / 4)
            unique_interactions = torch.unique(close_res, dim=0).to(int)
            if len(unique_interactions) != 0:
                interaction_pairs = torch.stack([stretch[unique_interactions[:, 0]], stretch[unique_interactions[:, 1]]], dim=1)
                valid_kmers = self._extract_valid_kmers(kmer_length, interaction_pairs)
                if valid_kmers is not None:
                    all_interactions.append(valid_kmers)

        if len(all_interactions) != 0:
            return torch.cat(all_interactions)

        return torch.Tensor([])

    
    def extract_interactions(self, filename: str, kmer_length: int, structured: bool=False, outname_stem: str=None, n_samples: int=10) -> torch.Tensor:
        """Extracts structured interactions"""
        coord = torch.cat(self.parse_all_chains(filename))
        all_interactions = self._find_interacting_motifs(coord, kmer_length, structured=structured)
        if len(all_interactions) == 0:
            return torch.Tensor([])
        strand_data = self.find_beta_strand_pairs(coord)
        bstrand_idx = strand_data[0] if strand_data is not None else torch.Tensor([])
        
        # remove interactions for which both residues are involved in beta strands
        new_shape = (all_interactions.shape[0], -1)
        if len(bstrand_idx) != 0:
            bstrand_interactions = torch.isin(all_interactions.reshape(new_shape), bstrand_idx)
        else:
            bstrand_interactions = all_interactions.reshape(new_shape).to(bool)
        possible_interactions = all_interactions[bstrand_interactions.sum(dim=1) < bstrand_interactions.shape[-1]//2]
        
        # remove interactions that are too close together in sequence space (residue separation determined by self.buffer)
        interaction_centers = possible_interactions.mean(dim=-1)
        final_mask = torch.absolute(interaction_centers[:, 0] - interaction_centers[:, 1]) > self.buffer
        final_interactions = possible_interactions[final_mask]
        
        if outname_stem is not None:
            atom_map = {0: 'N', 1: 'CA', 2: 'C', 3: 'O'}
            elem_map = {0: 'N', 1: 'C', 2: 'C', 3: 'O'}
            mask = np.random.choice(len(final_interactions), size=len(final_interactions) if n_samples == -1 else n_samples, replace=False)
            samples = final_interactions[mask].reshape(len(mask), -1)
            for i, interaction in enumerate(samples):
                atom_num = 1
                pdb_str = ''
                for res in sorted(interaction.to(int).tolist()):
                    c = coord[res]
                    for j, xyz in enumerate(c):
                        pdb_str += self.pdb_line(atom_num, atom_map[j], res+1, xyz, elem_map[j])
                        atom_num += 1
                
                with open(f'{outname_stem}_{i}.pdb', 'w') as f:
                    f.write(pdb_str)

        return final_interactions

class DihedralExtractor(StructUtils):
    """Extracts dihedrals from pdbs"""
    def __init__(self):
        super().__init__()
    
    def _prepare_dihedral_matrix(self, coord: torch.Tensor) -> torch.Tensor:
        """
        Constructs a matrix of shape [batch, partner, nres-2, 3, 3]
        Atom order for phi: C(-1), N, CA, C
        Atom order for psi: N, CA, C, N(+1)

        Returns a matrix of shape [2, n_residues, 4, 3]
        The 0th layer in the 0th dimension is the phi matrix.
        The 1st layer in the 0th dimension is the psi matrix.
        """
        nres = coord.shape[0]
        N_idx, CA_idx, C_idx = 0, 1, 2
        dihedral_matrix = torch.zeros(2, nres-2, 4, 3)

        # phi
        dihedral_matrix[0, :, 0] = coord[:-2, C_idx]
        dihedral_matrix[0, :, 1] = coord[1:-1, N_idx]
        dihedral_matrix[0, :, 2] = coord[1:-1, CA_idx]
        dihedral_matrix[0, :, 3] = coord[1:-1, C_idx]
        #psi
        dihedral_matrix[1, :, 0] = coord[1:-1, N_idx]
        dihedral_matrix[1, :, 1] = coord[1:-1, CA_idx]
        dihedral_matrix[1, :, 2] = coord[1:-1, C_idx]
        dihedral_matrix[1, :, 3] = coord[2:, N_idx]
        return dihedral_matrix

    def extract_dihedrals(self, pdb: str, chain: str, return_coord: bool=True) -> torch.Tensor:
        """Extracts phi and psi angles from a pdb chain"""
        coord = self.parse_chain(pdb, chain)
        main_chain_coord = coord[:, :-1]
        dihedral_matrix = self._prepare_dihedral_matrix(main_chain_coord)
        ab = dihedral_matrix[:, :, 0] - dihedral_matrix[:, :, 1]
        cb = dihedral_matrix[:, :, 2] - dihedral_matrix[:, :, 1]
        db = dihedral_matrix[:, :, 3] - dihedral_matrix[:, :, 2]
        
        u = torch.cross(ab, cb, axis=-1)
        v = torch.cross(db, cb, axis=-1)
        w = torch.cross(u, v, axis=-1)

        angles = self.calc_angle_batched(u, v)
        sign_mask = self.calc_angle_batched(cb, w) > 0.001
        angles[sign_mask] *= -1
        
        phi = angles[0]
        psi = angles[1]

        if return_coord:
            return coord, phi, psi
        return phi, psi

class DihedralDatabaseExtractor(StructUtils):
    """Used for extracting batched dihedrals from kmer pairs"""
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
    
    def _prepare_dihedral_matrix(self, coord: torch.Tensor) -> torch.Tensor:
        """
        Constructs a matrix of shape [batch, partner, nres-2, 3, 3]
        Atom order for phi: C(-1), N, CA, C
        Atom order for psi: N, CA, C, N(+1)

        Returns a matrix of shape [2, n_residues, 4, 3]
        The 0th layer in the 0th dimension is the phi matrix.
        The 1st layer in the 0th dimension is the psi matrix.
        """
        nres, _, _ = coord.shape
        N_idx, CA_idx, C_idx = 0, 1, 2
        dihedral_matrix = torch.zeros(2, nres-2, 4, 3)

        # phi
        dihedral_matrix[0, :, 0] = coord[:-2, C_idx]
        dihedral_matrix[0, :, 1] = coord[1:-1, N_idx]
        dihedral_matrix[0, :, 2] = coord[1:-1, CA_idx]
        dihedral_matrix[0, :, 3] = coord[1:-1, C_idx]
        #psi
        dihedral_matrix[1, :, 0] = coord[1:-1, N_idx]
        dihedral_matrix[1, :, 1] = coord[1:-1, CA_idx]
        dihedral_matrix[1, :, 2] = coord[1:-1, C_idx]
        dihedral_matrix[1, :, 3] = coord[2:, N_idx]
        return dihedral_matrix

    def extract_dihedral_pairs(self, pdb: str, kmer_length: int, residx1: torch.Tensor, residx2: torch.Tensor) -> tuple[torch.Tensor]:
        """Extacts dihedrals from a pdb at the given residue indices"""
        assert residx1.shape == residx2.shape
        bb_coord = self.parse_all_chains(os.path.join(self.data_dir, pdb))
        mainchain_coord = bb_coord[0][:, :-1] if len(bb_coord) == 1 else torch.cat(bb_coord, dim=0)[:, :-1]
        dihedral_matrix = self._prepare_dihedral_matrix(mainchain_coord)
        ab = dihedral_matrix[:, :, 0] - dihedral_matrix[:, :, 1]
        cb = dihedral_matrix[:, :, 2] - dihedral_matrix[:, :, 1]
        db = dihedral_matrix[:, :, 3] - dihedral_matrix[:, :, 2]
        
        u = torch.cross(ab, cb, axis=-1)
        v = torch.cross(db, cb, axis=-1)
        w = torch.cross(u, v, axis=-1)

        angles = self.calc_angle_batched(u, v) 
        angles[self.calc_angle_batched(cb, w) > 0.001] *= -1

        results = torch.zeros(residx1.shape[0], 2, 2*kmer_length)
        for i, (idx1, idx2) in enumerate(zip(residx1, residx2)):
            results[i, 0, :kmer_length] = angles[0, idx1-1]
            results[i, 0, kmer_length:] = angles[1, idx1-1]
            results[i, 1, :kmer_length] = angles[0, idx2-1]
            results[i, 1, kmer_length:] = angles[1, idx2-1]

        return results

    def __call__(self, pdb: str, kmer_length: int, residx1: torch.Tensor, residx2: torch.Tensor) -> tuple[torch.Tensor]:
        return self.extract_dihedral_pairs(pdb, kmer_length, residx1, residx2)
