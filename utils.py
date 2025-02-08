import numpy as np
import torch
from einops import repeat, rearrange
from Bio.PDB import PDBParser


class StructUtils:
    """
    Parses structures and performs pydssp assignments.
    pydssp scripts modified from https://github.com/ShintaroMinami/PyDSSP/blob/master/pydssp/pydssp_numpy.py
    """
    CONST_Q1Q2 = 0.084
    CONST_F = 332
    DEFAULT_CUTOFF = -0.5
    DEFAULT_MARGIN = 1.0
    parser = PDBParser(QUIET=True)
    atoms = ['N', 'CA', 'C', 'O']
    atom_set = set(atoms)
    
    def parse_chain(self, filename: str, chain: str) -> torch.Tensor:
        structure = self.parser.get_structure('chain', filename)[0][chain]
        coords = list()
        for residue in structure:
            current_atoms = set([atom.get_name() for atom in residue.get_atoms()])
            if len(self.atom_set - current_atoms) == 0:
                coords += [residue[atom].coord for atom in self.atoms]
        np_coords = np.asarray(coords).reshape((-1, 4, 3))
        return torch.from_numpy(np_coords)
    
    def parse_all_chains(self, filename: str) -> list:
        pdb_coords = list()
        for chain in self.parser.get_structure('complex', filename)[0]:
            coords = list()
            for residue in chain:
                current_atoms = [atom.get_name() for atom in residue.get_atoms()]
                if len(self.atom_set - set(current_atoms)) == 0:
                    coords += [residue[atom].coord for atom in self.atoms]
            np_coords = np.asarray(coords).reshape((-1, 4, 3))
            pdb_coords.append(torch.from_numpy(np_coords))
        return pdb_coords
    
    def generate_pseudo_cb(self, coord: torch.Tensor) -> torch.Tensor:
        """Calculates psuedo Cb coordinates for the given interactions"""
        N_xyz = coord[:, 0]
        CA_xyz = coord[:, 1]
        C_xyz = coord[:, 2]

        b = CA_xyz - N_xyz
        c = C_xyz - CA_xyz
        a = torch.cross(b, c, dim=-1)
        CB_xyz = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA_xyz
        CB_xyz = CB_xyz[:, None]
        full_coords = torch.cat([coord, CB_xyz], dim=1)
        return full_coords
    
    def calc_angle(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Calculates angle between two vectors"""
        vec1_normalized = vec1 / torch.norm(vec1)
        vec2_normalized = vec2 / torch.norm(vec2)
        dot_prod = (vec1_normalized * vec2_normalized).sum()
        return torch.arccos(dot_prod.clamp(-1.0, 1.0))

    def calc_angle_batched(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Calculates angles between multiple vectors stacked on the zeroeth dimension"""
        vec1 = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
        vec2 = vec2 / torch.norm(vec2, dim=-1, keepdim=True)
        angle = (vec1 * vec2).sum(dim=-1).clip(-1.0, 1.0)
        return torch.arccos(angle)

    def pdb_line(self, atom_num: int, atom_name: str, resnum: int, coord: torch.Tensor, elem: str, charge: str='') -> str:
        """General format for writing a pdb line"""
        return f'ATOM   {atom_num:>4}  {atom_name:<4}ALA A{resnum:>4}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00            A   {elem:>2}{charge}\n'
    
    def _check_input(self, coord: torch.Tensor) -> (torch.Tensor, tuple):
        org_shape = coord.shape
        assert (len(org_shape)==3) or (len(org_shape)==4), "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
        coord = repeat(coord, '... -> b ...', b=1) if len(org_shape)==3 else coord
        return coord, org_shape

    def _get_hydrogen_atom_position(self, coord: torch.Tensor) -> torch.Tensor:
        # A little bit lazy (but should be OK) definition of H position here.
        vec_cn = coord[:,1:,0] - coord[:,:-1,2]
        vec_cn = vec_cn / torch.linalg.norm(vec_cn, dim=-1, keepdim=True)
        vec_can = coord[:,1:,0] - coord[:,1:,1]
        vec_can = vec_can / torch.linalg.norm(vec_can, dim=-1, keepdim=True)
        vec_nh = vec_cn + vec_can
        vec_nh = vec_nh / torch.linalg.norm(vec_nh, dim=-1, keepdim=True)
        return coord[:,1:,0] + 1.01 * vec_nh

    def get_hbond_map(
        self,
        coord: torch.Tensor,
        donor_mask: torch.Tensor=None,
        cutoff: float=DEFAULT_CUTOFF,
        margin: float=DEFAULT_MARGIN,
        return_e: bool=False
        ) -> torch.Tensor:
        # check input
        coord, org_shape = self._check_input(coord)
        b, l, a, _ = coord.shape
        # add pseudo-H atom if not available
        assert (a==4) or (a==5), "Number of atoms should be 4 (N,CA,C,O) or 5 (N,CA,C,O,H)"
        h = coord[:,1:,4] if a == 5 else self._get_hydrogen_atom_position(coord)
        # distance matrix
        nmap = repeat(coord[:,1:,0], '... m c -> ... m n c', n=l-1)
        hmap = repeat(h, '... m c -> ... m n c', n=l-1)
        cmap = repeat(coord[:,0:-1,2], '... n c -> ... m n c', m=l-1)
        omap = repeat(coord[:,0:-1,3], '... n c -> ... m n c', m=l-1)
        d_on = torch.linalg.norm(omap - nmap, dim=-1)
        d_ch = torch.linalg.norm(cmap - hmap, dim=-1)
        d_oh = torch.linalg.norm(omap - hmap, dim=-1)
        d_cn = torch.linalg.norm(cmap - nmap, dim=-1)
        # electrostatic interaction energy
        e = torch.nn.functional.pad(self.CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn)*self.CONST_F, [0,1,1,0])
        if return_e: return e
        # mask for local pairs (i,i), (i,i+1), (i,i+2)
        local_mask = ~torch.eye(l, dtype=bool)
        local_mask *= ~torch.diag(torch.ones(l-1, dtype=bool), diagonal=-1)
        local_mask *= ~torch.diag(torch.ones(l-2, dtype=bool), diagonal=-2)
        # mask for donor H absence (Proline)
        if donor_mask is None:
            donor_mask = torch.ones(l, dtype=float)
        else:
            donor_mask = donor_mask.to(float) if torch.is_tensor(donor_mask) else torch.Tensor(donor_mask).to(float)
        donor_mask = repeat(donor_mask, 'l1 -> l1 l2', l2=l)
        # hydrogen bond map (continuous value extension of original definition)
        hbond_map = torch.clamp(self.DEFAULT_CUTOFF - self.DEFAULT_MARGIN - e, min=-self.DEFAULT_MARGIN, max=self.DEFAULT_MARGIN)
        hbond_map = (torch.sin(hbond_map/self.DEFAULT_MARGIN*torch.pi/2)+1.)/2
        hbond_map = hbond_map * repeat(local_mask.to(hbond_map.device), 'l1 l2 -> b l1 l2', b=b)
        hbond_map = hbond_map * repeat(donor_mask.to(hbond_map.device), 'l1 l2 -> b l1 l2', b=b)
        # return h-bond map
        hbond_map = hbond_map.squeeze(0) if len(org_shape)==3 else hbond_map
        return hbond_map

    def assign(self, coord: torch.Tensor, donor_mask: torch.Tensor=None) -> torch.Tensor:
        # check input
        coord, org_shape = self._check_input(coord)
        # get hydrogen bond map
        original_hbmap = self.get_hbond_map(coord, donor_mask=donor_mask)
        hbmap = rearrange(original_hbmap, '... l1 l2 -> ... l2 l1') # convert into "i:C=O, j:N-H" form
        # identify turn 3, 4, 5
        turn3 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=3) > 0.
        turn4 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=4) > 0.
        turn5 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=5) > 0.
        # assignment of helical sses
        h3 = torch.nn.functional.pad(turn3[:,:-1] * turn3[:,1:], [1,3])
        h4 = torch.nn.functional.pad(turn4[:,:-1] * turn4[:,1:], [1,4])
        h5 = torch.nn.functional.pad(turn5[:,:-1] * turn5[:,1:], [1,5])
        # helix4 first
        helix4 = h4 + torch.roll(h4, 1, 1) + torch.roll(h4, 2, 1) + torch.roll(h4, 3, 1)
        h3 = h3 * ~torch.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
        h5 = h5 * ~torch.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
        helix3 = h3 + torch.roll(h3, 1, 1) + torch.roll(h3, 2, 1)
        helix5 = h5 + torch.roll(h5, 1, 1) + torch.roll(h5, 2, 1) + torch.roll(h5, 3, 1) + torch.roll(h5, 4, 1)
        # identify bridge
        unfoldmap = hbmap.unfold(-2, 3, 1).unfold(-2, 3, 1) > 0.
        unfoldmap_rev = unfoldmap.transpose(-4,-3)
        p_bridge = (unfoldmap[:,:,:,0,1] * unfoldmap_rev[:,:,:,1,2]) + (unfoldmap_rev[:,:,:,0,1] * unfoldmap[:,:,:,1,2])
        p_bridge = torch.nn.functional.pad(p_bridge, [1,1,1,1])
        a_bridge = (unfoldmap[:,:,:,1,1] * unfoldmap_rev[:,:,:,1,1]) + (unfoldmap[:,:,:,0,2] * unfoldmap_rev[:,:,:,0,2])
        a_bridge = torch.nn.functional.pad(a_bridge, [1,1,1,1])
        # ladder
        ladder = (p_bridge + a_bridge).sum(-1) > 0
        # H, E, L of C3
        helix = (helix3 + helix4 + helix5) > 0
        strand = ladder
        loop = (~helix * ~strand)
        onehot = torch.stack([loop, helix, strand], dim=-1)
        onehot = onehot.squeeze(0) if len(org_shape)==3 else onehot
        return onehot, original_hbmap[0]
