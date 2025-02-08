import os
import random
import numpy as np
import torch
from typing import Union
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm

import extractor_models as extractor

class StrandWriter(extractor.StrandExtractor):
    """Write beta strand residues into a pdb"""
    def __init__(
        self,
        interactions: str,
        nsamples: int=15000,
        include_filters: bool=True,
        clash_cutoff: float=3,
        cluster_threshold: float=1.0,
        only_cacb: bool=False
    ):
        super().__init__()
        self.interactions = torch.load(interactions)
        self.nsamples = nsamples
        self.include_filters = include_filters
        self.cutoff = clash_cutoff
        self.cluster_threshold = cluster_threshold
        self.only_cacb = only_cacb
    
    def parse_pdb(self, filename: str, chain: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Parses the chain and return the backbone coordinates and full atom coordinates"""
        bb_coord = self.parse_chain(filename, chain)
        fa_coord = list()
        for residue in self.parser.get_structure('fa', filename)[0][chain]:
            for atom in residue:
                if atom.element != 'H':
                    fa_coord.append(atom.coord)
        fa_coord = torch.from_numpy(np.asarray(fa_coord))
        return bb_coord, fa_coord
    
    def perform_transformations(self, bb_coord: torch.Tensor, fa_coord: torch.Tensor, residx: Union[torch.Tensor, list]) -> Union[torch.Tensor, torch.Tensor]:
        """Calculates coordinates of interacting beta strand residues with respect to residues specified by residx"""
        # collect transformation matrices
        if self.nsamples == 0:
            trans = self.interactions['trans']
            rot = self.interactions['rot']
        else:
            chosen = random.sample(range(len(self.interactions['trans'])), self.nsamples)
            trans = self.interactions['trans'][chosen]
            rot = self.interactions['rot'][chosen]

        # perform rototranslation on relevant residues
        residx = torch.Tensor(residx).to(int) if isinstance(residx, list) else residx
        expanded_coord = bb_coord[residx].reshape((-1, 1, 4, 3))
        centroids = expanded_coord.mean(dim=2, keepdims=True)
        interactions = ((expanded_coord - centroids) @ rot.transpose(2, 1) + centroids + trans[:, None]).reshape(-1, 4, 3)

        if self.include_filters:
            valid_hbond_interactions = self._bstrand_Hbond_check(bb_coord, interactions, residx)
            clashes_removed = self._remove_interacting_clashes(fa_coord, valid_hbond_interactions)
            clustered = self._cluster_strand_interactions(clashes_removed)
            target_residues = residx[self._get_interacting_residues(bb_coord, clustered, residx)[1][:, 1]]
            return clustered, target_residues

        return interactions, residx[self._get_interacting_residues(interactions, residx)[1][:, 1]]
    
    def _bstrand_Hbond_check(self, bb_coord: torch.Tensor, interactions: Union[list, torch.Tensor], residx: Union[list, torch.Tensor]) -> torch.Tensor:
        """Checks that the relative atomic distances and bond angles are appropriate for Hbonding"""
        residx = torch.Tensor(residx).to(int) if isinstance(residx, list) else residx
        hbond_distances, valid_pairs = self._get_interacting_residues(bb_coord, interactions, residx)
        unique, counts = torch.unique(valid_pairs[:, 0], return_counts=True)
        invalid_interactions = set(unique[torch.nonzero(counts > 1, as_tuple=True)].tolist())
        distance_mask = [i for i in torch.unique(valid_pairs[:, 0]).tolist() if i not in invalid_interactions]
        updated_interactions = interactions[distance_mask]

        # check CO angles
        target_partner_idx = torch.argwhere(hbond_distances[distance_mask])[:, 1]
        current_residx = residx[target_partner_idx]
        CO_vectors = torch.stack(
            [updated_interactions[:, 3] - updated_interactions[:, 2], 
            bb_coord[current_residx, 3] - bb_coord[current_residx, 2]],
            dim=1
        )
        CO_vectors /= torch.norm(CO_vectors, dim=-1, keepdim=True)
        CO_angle_matrix = torch.arccos((CO_vectors[:, 0] * CO_vectors[:, 1]).sum(dim=-1).clamp(-1.0, 1.0))
        CO_angle_mask = torch.argwhere(CO_angle_matrix > torch.pi - torch.pi / 8).flatten()
        updated_interactions = updated_interactions[CO_angle_mask]

        # check NC angles (assumes antiparallel)
        current_residx = current_residx[CO_angle_mask]
        CN_vectors = torch.stack(
            [updated_interactions[:, 2] - updated_interactions[:, 0],
            bb_coord[current_residx, 2] - bb_coord[current_residx, 0]],
            dim=1
        )
        CN_vectors /= torch.norm(CN_vectors, dim=-1, keepdim=True)
        CN_angle_matrix = torch.arccos((CN_vectors[:, 0] * CN_vectors[:, 1]).sum(dim=-1).clamp(-1.0, 1.0))
        CN_angle_mask = torch.argwhere(CN_angle_matrix > torch.pi - torch.pi / 8).flatten()
        final_interactions = updated_interactions[CN_angle_mask]

        return final_interactions
    
    def _remove_interacting_clashes(self, fa_coord: torch.Tensor, interactions: torch.Tensor) -> torch.Tensor:
        """Remove clashing interaction based on centroid distances"""        
        interaction_with_cb = self.generate_pseudo_cb(interactions).reshape(-1, 1, 3)
        dm = torch.linalg.norm(interaction_with_cb - fa_coord[None], dim=-1)
        atom_clashes = torch.floor(torch.argwhere(dm < self.cutoff)[:, 0] / 5).to(int)
        to_remove = set(atom_clashes.tolist())
        invert = [i for i in range(len(interactions)) if i not in to_remove]
        return interactions[invert]

    def _cluster_strand_interactions(self, interactions: torch.Tensor) -> torch.Tensor:
        """Uses agglomerative hierarchical clustering to prune the matches"""
        assert 1==2, 'rewrite clustering algorithm with scipy implementation'
        if self.cluster_threshold == 0:
            return interactions

        clusterer = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=self.cluster_threshold)
        clusters = clusterer.fit_predict(interactions.reshape(interactions.shape[0], -1))

        cluster_mask = list()
        for i in np.unique(clusters):
            random_member = random.choice(np.argwhere(clusters == i).flatten())
            cluster_mask.append(random_member)

        return interactions[cluster_mask]
    
    def _get_interacting_residues(self, bb_coord: torch.Tensor, interactions: torch.Tensor, residx: Union[torch.Tensor, list]) -> torch.Tensor:
        """For each interaction, identifies its partner residue on the target"""
        interacting_atoms = torch.stack([interactions[:, 0], interactions[:, 3]], dim=1)
        target_atoms = torch.stack([bb_coord[residx, 3], bb_coord[residx, 0]], dim=1)
        length_matrix = torch.norm(interacting_atoms[:, None] - target_atoms[None], dim=-1)
        hbond_distances = torch.all(torch.logical_and(length_matrix > 2.7, length_matrix < 3.2), dim=-1)
        valid_pairs = torch.argwhere(hbond_distances)
        return hbond_distances, valid_pairs
    
    def write_interactions(self, filename: str, chain: str, residx: list=[], outname: str='') -> Union[torch.Tensor, torch.Tensor]:
        """Calculates and optionally writes interactions to a pdb"""
        bb_coord, fa_coord = self.parse_pdb(filename, chain)
        residx = residx if len(residx) != 0 else list(range(len(bb_coord)))
        interactions, interacting_pairs = self.perform_transformations(bb_coord, fa_coord, residx)
        full_bb_interactions = self.generate_pseudo_cb(interactions)

        if outname != '':
            pdb_str = ''
            atom_num = 1
            interactions = interactions.reshape(-1, 4, 3)
            if self.only_cacb:
                for i, interaction in enumerate(full_bb_interactions):
                    pdb_str += self.pdb_line(atom_num, 'CA', i+1, interaction[1], 'C')
                    atom_num += 1
                    pdb_str += self.pdb_line(atom_num, 'CB', i+1, interaction[4], 'C')
                    atom_num += 1
                    
            else:
                atom_map = {0: 'N', 1: 'CA', 2: 'C', 3: 'O', 4: 'CB'}
                elem_map = {0: 'N', 1: 'C', 2: 'C', 3: 'O', 4: 'C'}
                for i, interaction in enumerate(full_bb_interactions):
                    for j, coord in enumerate(interaction):
                        pdb_str += self.pdb_line(atom_num, atom_map[j], i+1, coord, elem_map[j])
                        atom_num += 1

            with open(outname, 'w') as f:
                f.write(pdb_str)
        
        return full_bb_interactions, interacting_pairs

class KMerWriter(extractor.DihedralExtractor):
    def __init__(
        self,
        interactions: str,
        data_dir: str='/scratch/groups/possu/cath/pdb_store',
        match_params: dict={3: 0.03, 4: 0.04, 5: 0.05},
        clash_cutoff=2.5
    ):
        super().__init__()
        self.interactions = torch.load(interactions)
        self.data_dir = data_dir
        self.match_params = match_params
        self.clash_cutoff = clash_cutoff

    def _parse_structure(self, pdb: str, chain: str, residx: list) -> dict:
        """Extracts coords and dihedrals of structured motifs"""
        data = self.extract_dihedrals(pdb, chain, return_coord=True)
        bb_coord = data[0][residx].to(torch.float)
        phi, psi = data[1][[i-1 for i in residx]], data[2][[i-1 for i in residx]]

        fa_coord = list()
        for residue in self.parser.get_structure('fa', pdb)[0][chain]:
            for atom in residue:
                fa_coord.append(atom.coord)
        fa_coord = torch.from_numpy(np.asarray(fa_coord))

        secondary_struct, _ = self.assign(data[0])
        structured_residx = torch.argwhere(torch.logical_or(secondary_struct[residx, 1], secondary_struct[residx, 2])).flatten()
        structured_stretches = self._extract_structured_stretches(structured_residx)

        # collect kmer dihedrals
        three_mer_residx, three_mer_dihedrals = self._collect_kmer_dihedrals(structured_stretches, 3, phi, psi)
        four_mer_residx, four_mer_dihedrals = self._collect_kmer_dihedrals(structured_stretches, 4, phi, psi)
        five_mer_residx, five_mer_dihedrals = self._collect_kmer_dihedrals(structured_stretches, 5, phi, psi)
        kmer_data = {
            3: {'dihedral_residx': three_mer_residx, 'dihedrals': three_mer_dihedrals},
            4: {'dihedral_residx': four_mer_residx, 'dihedrals': four_mer_dihedrals},
            5: {'dihedral_residx': five_mer_residx, 'dihedrals': five_mer_dihedrals}
        }
        return bb_coord, fa_coord, kmer_data
    
    def _extract_structured_stretches(self, structured_residx: torch.Tensor) -> list[torch.Tensor]:
        """Extracts stretches of contiguous structured domains"""
        residx = structured_residx.tolist()
        stretch_counter, residx_counter = 0, 0
        results = {stretch_counter: [residx[0]]}
        for delta in torch.diff(structured_residx):
            residx_counter += 1
            if delta != 1:
                stretch_counter += 1
                results[stretch_counter] = list()
            results[stretch_counter].append(residx[residx_counter])
        return [v for v in results.values()]
    
    def _collect_kmer_dihedrals(self, residx_stretch: list[list], kmer_length: int, phi: torch.Tensor, psi: torch.Tensor) -> tuple[torch.Tensor]:
        """Collects the residx of kmers from the structured stretches"""
        fragments = list()
        for stretch in residx_stretch:
            if len(stretch) < kmer_length:
                continue

            counter = 0
            while counter <= len(stretch) - kmer_length:
                fragments.append(stretch[counter:counter+kmer_length])
                counter += 1

        if len(fragments) == 0:
            return torch.Tensor([]), torch.Tensor([])

        # assemble kmer residx
        kmers = torch.Tensor(fragments).to(int)
        kmers = kmers[torch.all(kmers > 0, dim=1)] # removes any kmers that include first residue
        kmers_shifted = kmers - 1 # corrects off-by-one error for indexing into dihedral tensor
        kmer_start_idx = kmers_shifted[:, 0].tolist()

        # accumulate dihedral_angles
        result = torch.zeros(kmers_shifted.shape[0], 2 * kmer_length)
        for i, mer in enumerate(kmers_shifted):
            result[i, :kmer_length] = phi[mer]
            result[i, kmer_length:] = psi[mer]
        return kmer_start_idx, result

    def _angular_dm(self, target_dihedrals: torch.Tensor, database_dihedrals: torch.Tensor) -> torch.Tensor:
        """Uses trig identity to more efficiently calculate angular norm"""
        return torch.sqrt((2 - 2 * torch.cos(target_dihedrals[:, None] - database_dihedrals[None])).sum(dim=-1) / target_dihedrals.shape[-1])

    def _find_kmer_matches(self, parsed_dihedrals: dict, kmer_length: Union[int, list]) -> dict[int,dict]:
        """Finds the indices of matching kmers"""
        kmer_length = [kmer_length] if isinstance(kmer_length, int) else kmer_length
        results = dict()
        for kmer in kmer_length:
            assert kmer in {3, 4, 5}, "Currently only accepts 3-, 4-, and 5-mers"
            if len(parsed_dihedrals[kmer]['dihedral_residx']) == 0:
                continue

            target_dihedrals = parsed_dihedrals[kmer]['dihedrals']
            forward_matches = self._angular_dm(target_dihedrals, self.interactions['dihedrals'][kmer]['forward'])
            reverse_matches = self._angular_dm(target_dihedrals, self.interactions['dihedrals'][kmer]['reverse'])

            if len(forward_matches) != 0 or len(reverse_matches) != 0:
                results[kmer] = {'forward': dict(), 'reverse': dict()}
                
                match_cutoff = self.match_params[kmer]
                for residx_f, interaction_f in torch.argwhere(forward_matches < match_cutoff).tolist():
                    if residx_f+1 not in results[kmer]['forward']:
                        results[kmer]['forward'][residx_f+1] = list()
                    results[kmer]['forward'][residx_f+1].append(interaction_f)

                for residx_r, interaction_r in torch.argwhere(reverse_matches < match_cutoff).tolist():
                    if residx_r+1 not in results[kmer]['reverse']:
                        results[kmer]['reverse'][residx_r+1] = list()
                    results[kmer]['reverse'][residx_r+1].append(interaction_r)

        return results

    def _construct_pdb_lookup_table(self, kmer_length: int) -> dict[int,str]:
        """Constructs a lookup table whose key is the ith interaction and value is the pdb"""
        i = 0
        results = dict()
        for name, (start, end) in self.interactions['pdbs'][kmer_length].items():
            for _ in range(end-start):
                results[i] = name
                i += 1

        return results

    def _extract_interaction_coordinates(self, compiled_searches: dict) -> dict[str,dict]:
        """Main function for iterating through all dataset structures and extracting relevant coordinates"""
        results = dict()
        for pdb, interactions in tqdm(compiled_searches.items()):
            filename = os.path.join(self.data_dir, pdb)
            parsed_coord = self.parse_all_chains(filename)
            full_coord = parsed_coord[0] if len(parsed_coord) == 1 else torch.cat(parsed_coord, dim=0)
            for template_residx, query_residx in zip(interactions['interactions'], interactions['residx']):
                paired_coord = torch.stack([full_coord[template_residx[0]], full_coord[template_residx[1]]], dim=0)
                if paired_coord.shape[1] not in results:
                    results[paired_coord.shape[1]] = {'coord': list(), 'residx': list()}
                results[paired_coord.shape[1]]['coord'].append(paired_coord)
                results[paired_coord.shape[1]]['residx'].append(query_residx)
        
        for mer in results:
            results[mer] = (
                torch.stack(results[mer]['coord'], dim=0).to(torch.float),
                torch.Tensor(results[mer]['residx']).to(int),
            )
        
        return results

    def _generate_fragments(self, extracted_interaction_coord: dict[int,tuple[torch.Tensor]], target_coord: torch.Tensor) -> dict[int,torch.Tensor]:
        """Uses Kabsch to calculate rotation and translation matrices between database coordinates and query coordinates"""
        results = dict()
        for k, (interactions, residx) in extracted_interaction_coord.items():
            sample = interactions[:, 0]
            template = torch.stack([target_coord[i:i+k] for i in residx], dim=0)
            T, R = self._kabsch(sample, template)
            fragments = interactions[:, 1].reshape(interactions.shape[0], k*4, 3)
            fragments_centroid = fragments.mean(dim=1, keepdim=True)
            aligned_fragments = ((fragments - fragments_centroid) @ R.transpose(1,2) + fragments_centroid + T).reshape(fragments.shape[0], k, 4, 3)
            results[k] = aligned_fragments
        return results
    
    def _kabsch(self, sample: torch.Tensor, template: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates batched rotation and translation matrices to transform sample -> template"""
        assert sample.shape == template.shape, 'Tensor dimensions must match'
        sample_reshaped, template_reshaped = sample.reshape(sample.shape[0], -1, 3), template.reshape(template.shape[0], -1, 3)
        sample_centroid, template_centroid = sample_reshaped.mean(dim=1, keepdim=True), template_reshaped.mean(dim=1, keepdim=True)
        opt_T = template_centroid - sample_centroid
        sample_centered = sample_reshaped - sample_centroid
        template_centered = template_reshaped - template_centroid
        H = sample_centered.transpose(1, 2) @ template_centered
        U, _, Vt = torch.linalg.svd(H)
        Vt[torch.linalg.det(Vt.transpose(1,2) @ U.transpose(1,2)) < 0.0, -1] *= -1.0
        opt_R = Vt.transpose(1,2) @ U.transpose(1,2)
        return opt_T, opt_R
    
    def _remove_clashing_fragments(self, fragments: dict[int,torch.Tensor], fa_target_coord: torch.Tensor) -> dict[int,torch.Tensor]:
        """Removes clashing fragments based on a distance cutoff"""
        results = dict()
        for k, frag in fragments.items():
            frag_with_cb = self.generate_pseudo_cb(frag.reshape(-1, 4, 3)).reshape(-1, k, 5, 3)
            distance_matrix = torch.norm(frag_with_cb.reshape(frag.shape[0], -1, 1, 3) - fa_target_coord[None], dim=-1)
            clashes = torch.any(distance_matrix.reshape(frag.shape[0], -1) < self.clash_cutoff, dim=1)
            clash_idx = torch.argwhere(clashes).flatten()
            remaining_interactions = frag_with_cb[[i for i in range(frag.shape[0]) if i not in clash_idx]]
            results[k] = remaining_interactions
        return results

    def _cluster_interactions(self, interactions: dict, rmsd_threshold: float=0.5) -> dict:
        """Uses agglomerative clustering to remove fragment matches that are similar"""
        results = dict()
        for k, interaction in tqdm(interactions.items()):
            if len(interaction) == 0:
                continue
            elif interaction.shape[0] == 1:
                results[k] = interaction
                continue

            reduced_data = interaction[:, :, :-2].reshape(interaction.shape[0], -1)
            triu_idx = torch.triu_indices(reduced_data.shape[0], reduced_data.shape[0], 1)
            rmsd = torch.sqrt((reduced_data[triu_idx[0]] - reduced_data[triu_idx[1]]).pow(2).sum(dim=-1) / (k*3))
            Z = linkage(rmsd, method='ward')
            clusters = fcluster(Z, rmsd_threshold, criterion='distance')

            results[k] = list()
            for label in np.unique(clusters):
                mask = clusters == label
                n_members = mask.sum()
                if n_members == 1:
                    results[k].append(interaction[mask][0])
                elif n_members == 2:
                    random_idx = random.choice(np.argwhere(mask).flatten().tolist())
                    results[k].append(interaction[random_idx])
                else:
                    rmsd_sum = torch.sqrt((reduced_data[mask, None] - reduced_data[None, mask]).pow(2).sum(dim=-1) / (k*3)).sum(dim=-1)
                    results[k].append(interaction[torch.argmin(rmsd_sum)])

        return {k: torch.cat(v, dim=0).reshape(-1, k, 5, 3) if isinstance(v, list) else v for k, v in results.items()}

    def check_number_matches(self, pdb: str, chain: str, residx: list, three_mer_cutoff: float, four_mer_cutoff: float, five_mer_cutoff: float) -> dict:
        """Checks the number of matches for given distance cutoffs"""
        zero_indexed = [i-1 for i in residx]
        _, _, dihedrals = self._parse_structure(pdb, chain, zero_indexed)
        results = dict()
        for kmer in tqdm(range(3,6)):           
            target_dihedrals = dihedrals[kmer]['dihedrals']
            forward_matches = self._angular_dm(target_dihedrals, self.interactions['dihedrals'][kmer]['forward']).flatten()
            reverse_matches = self._angular_dm(target_dihedrals, self.interactions['dihedrals'][kmer]['reverse']).flatten()
            cutoff = three_mer_cutoff if kmer == 3 else four_mer_cutoff if kmer == 4 else five_mer_cutoff
            results[kmer] = (((forward_matches < cutoff).sum() + (reverse_matches < cutoff).sum()).item(), 2 * forward_matches.shape[0])
        return results

    def compile_interactions(self, pdb: str, chain: str, residx: list, kmer_lengths: Union[int, list[int]], outname: str) -> None:
        """Main function for identifying and optionally writing interactions to a pdb"""
        print('Finding matches!')
        zero_indexed = [i-1 for i in residx]
        _, _, dihedrals = self._parse_structure(pdb, chain, zero_indexed)
        kmer_matches = self._find_kmer_matches(dihedrals, kmer_lengths)
        torch.save(kmer_matches, outname)
        return

    def prepare_interaction_lookup(self, kmer_matches: dict) -> dict:
        """Compiles the matching dataset interactions to enable efficient reading of dataset coordinates"""
        to_parse = dict() # first parse kmer matches to efficiently read coordinates from multiple files
        for mer, match_d in kmer_matches.items():
            lookup_table = self._construct_pdb_lookup_table(mer)
            kmer_interactions = self.interactions['interactions'][mer].sort(dim=-1)[0]
            for direction, matches in match_d.items():
                for residx, match_sublst in matches.items():
                    for match_idx in match_sublst:
                        pdb_name = lookup_table[match_idx]
                        if direction == 'forward':
                            interaction = kmer_interactions[match_idx]
                        else: 
                            interaction = kmer_interactions[match_idx].flip(0)

                        if pdb_name not in to_parse:
                            to_parse[pdb_name] = {'interactions': list(), 'residx': list()}
                        to_parse[pdb_name]['interactions'].append(interaction)
                        to_parse[pdb_name]['residx'].append(residx)

        for k in to_parse:
            to_parse[k]['interactions'] = to_parse[k]['interactions']
            to_parse[k]['residx'] = torch.Tensor(to_parse[k]['residx'])

        return to_parse

    def parse_interaction_coordinates(self, pdb: str, chain: str, residx: list, compiled_interaction_idx: dict, outname: str) -> None:
        """Main function for parsing coordinates"""
        print('Parsing dataset coordinates!')
        zero_indexed = [i-1 for i in residx]
        bb_coord, fa_coord, _ = self._parse_structure(pdb, chain, zero_indexed)
        interaction_coord = self._extract_interaction_coordinates(compiled_interaction_idx)
        all_fragments = self._generate_fragments(interaction_coord, bb_coord)
        filtered_fragments = self._remove_clashing_fragments(all_fragments, fa_coord)
        clustered_fragments = self._cluster_interactions(filtered_fragments)
        torch.save(clustered_fragments, outname)
        return

    def write_pdb(self, interactions: torch.Tensor, outname: str) -> None:
            """Writes the kmer interactions to a pdb"""
            atom_map = {i: atom for i, atom in enumerate(['N', 'CA', 'C', 'O', 'CB'])}
            elem_map = {i: elem for i, elem in enumerate(['N', 'C', 'C', 'O', 'C'])}
            pdb_str = ''
            atom_num, residue = 1, 1
            for interaction in interactions:
                for fragment in interaction:
                    for i, atom_coord in enumerate(fragment):
                        pdb_str += self.pdb_line(atom_num, atom_map[i], residue, atom_coord, elem_map[i])
                        atom_num += 1
                    residue += 1
                with open(outname, 'w') as f:
                    f.write(pdb_str)
            return
