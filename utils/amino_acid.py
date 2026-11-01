from rdkit import Chem
import os, wget
import numpy as np
import utils.align_utils as autils
import utils.data_constants as ud_constants
from gemmi import cif
import wget
import torch

# Directory holding one <RESIDUE>.npy atom-embedding file per residue type.
# Overridden by set_atom_embed_dir() or the ATOM_EMBED_DIR environment variable.
ATOM_EMBED_DIR = os.environ.get(
    "ATOM_EMBED_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "aa_embed"))


def set_atom_embed_dir(path: str) -> None:
    """Point the loader at a different atom-embedding directory."""
    global ATOM_EMBED_DIR
    ATOM_EMBED_DIR = path


# Optional residue substitution, used only by the input-ablation experiments.
# None means no substitution, which is the behaviour of the published pipeline.
SUBSTITUTION = None



def set_substitution(mapping) -> None:
    """Replace each residue by mapping[residue] when building its atom graph.

    The original residue name is kept on the instance as `original_aa_name`, and the
    structure and its atom names are untouched, so a substituted residue takes the
    coordinates of whichever of its atoms exist in the deposited residue and silently
    drops the rest.
    """
    global SUBSTITUTION
    SUBSTITUTION = mapping



class AminoAcid():
    def __init__(
        self,
        aa_name: str,
        aa_idx: int,
        structure: object,
        ligand_path: str,
        include_default_oh_as_exist_atom: bool = False,
        is_missing: bool = False,
        init_exist_atoms: set = None
    ) -> None:
        self.ligand_path = ligand_path

        self.original_aa_name = aa_name
        target = (SUBSTITUTION.get(aa_name, aa_name)
                  if SUBSTITUTION is not None else aa_name)
        if target != aa_name:
            # A substituted residue stands in for a different chemistry, so it
            # contributes the replacement's template and atom embedding rather than
            # the geometry of the residue it replaces. It is therefore built as a
            # residue without coordinates and joined to its neighbours by the peptide
            # bond reconstructed from the templates, which is the path this pipeline
            # already uses for residues absent from the structure. Coordinates never
            # reach the network -- they serve only to recover connectivity -- so this
            # withholds nothing the model would otherwise see, and it avoids matching
            # atom names between residues that may share none.
            structure = None
            init_exist_atoms = None
            is_missing = True
        self.aa_name = target
        self.index = aa_idx
        self.include_default_oh_as_exist_atom = include_default_oh_as_exist_atom
        self.init_exist_atoms = init_exist_atoms
        self.is_missing = is_missing

        self._exist_atoms = []
        self._pseudo_backbones = []

        self.init_from_cif_and_sdf_files(structure=structure)
    
    # def _load_pre_emb(self):

    #      # path = os.path.join('/ibex/user/weil/PPI_NonAA/graphmae_emb/', self.aa_name+".pth")

    #     if self.aa_name in ["HOA", "NME", "NH2", "GLX", "MOH", "CF0", "0QE"]:
    #         path = os.path.join('/ibex/user/weil/PPI_NonAA/aa_emb/', self.aa_name+".pth")
    #         ret= torch.load(path)
    #         assert len(ret["node_emb"])==len(self.mol.GetAtoms()), f"{self.aa_name} {ret['node_emb'].shape} {len(self.mol.GetAtoms())}"
    #         out_fetch_idx=sorted(self._map_from_inner_idx_to_output_idx.keys(),key=lambda x:self._map_from_inner_idx_to_output_idx[x])
    #         ret["node_emb"]=ret["node_emb"][out_fetch_idx]

    #         assert len(ret["node_emb"]) == len(self._node_int_feats), self.aa_name
    #         return ret
        
    #     else:
    #         path = os.path.join('/ibex/user/rux/PPI_NonAA/aa_emb/', self.aa_name+".pth")

    #         ret= torch.load(path)
    #         # select node emb
    #         # assert len(ret["node_emb"])==len(self.mol.GetAtoms()), f"{self.aa_name} {ret['node_emb'].shape} {len(self.mol.GetAtoms())}"
    #         # out_fetch_idx=sorted(self._map_from_inner_idx_to_output_idx.keys(),key=lambda x:self._map_from_inner_idx_to_output_idx[x])
    #         ret_map_from_inner_idx_to_output_idx = ret["_map_from_inner_idx_to_output_idx"]
    #         aa_emb = np.zeros((self.non_H_atoms, 128))
    #         for key in self._map_from_inner_idx_to_output_idx.keys():
    #             aa_emb[self._map_from_inner_idx_to_output_idx[key]] = ret["node_emb"][ret_map_from_inner_idx_to_output_idx[key]]
            
    #         ret.pop("_map_from_inner_idx_to_output_idx")
    #         ret["node_emb"] = aa_emb
    #         assert len(ret["node_emb"]) == len(self._node_int_feats), f"{self.aa_name} {ret['node_emb'].shape} {self.non_H_atoms}"
    #         return ret

        # print('Error for loading pre-emb for ', self.aa_name)
        # return {}
    

    def _load_pre_emb(self):
        """Attach the pre-computed atom embedding for this residue type.

        The embedding is stored per residue type, so a residue contributes the same
        atom embeddings wherever it occurs. It is reordered from the template's atom
        order to this residue's output atom order.
        """
        path = os.path.join(ATOM_EMBED_DIR, self.aa_name + ".npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"atom embedding not found for {self.aa_name}: {path}")

        node_emb = torch.from_numpy(np.load(path).astype(np.float32, copy=False))
        assert len(node_emb) == len(self.mol.GetAtoms()), self.aa_name
        out_fetch_idx = sorted(self._map_from_inner_idx_to_output_idx.keys(),
                               key=lambda x: self._map_from_inner_idx_to_output_idx[x])
        node_emb = node_emb[out_fetch_idx]
        assert len(node_emb) == len(self._node_int_feats), self.aa_name
        return {"node_emb": node_emb}

    def _load_from_sdf(self, structure):
        sdf_path = os.path.join(self.ligand_path, f"{self.aa_name}_ideal.sdf")
        if not os.path.exists(
                sdf_path):
            wget.download(
                "https://files.rcsb.org/ligands/download/{}_ideal.sdf".format(self.aa_name), sdf_path)

        # load mol from sdf "HOA", "NME", "NH2", "GLX", "MOH", "CF0", "0QE"
        self.mol = Chem.MolFromMolFile(
            sdf_path,  sanitize=True, removeHs=False)

        if self.mol is None or self.aa_name in ["UNL", "U5G", "BVA"]:
            self.is_missing = True
            self.aa_name = ud_constants.WildCardAA
            # print("Warning: use wild card to replace unknown for ", sdf_path)
            # Fall back to the local GLY template for ligands such as GLX whose
            # SDF contains unresolved placeholder atoms (for example XE1/XE2).
            sdf_path = os.path.join(
                self.ligand_path, f"{self.aa_name}_ideal.sdf")
            self.mol = Chem.MolFromMolFile(
                sdf_path,  sanitize=True, removeHs=False)
            structure = None
        return structure

    def _load_from_cif(self):
        cif_path = os.path.join(self.ligand_path, f"{self.aa_name}.cif")
        if not os.path.exists(
                cif_path):
            wget.download(
                "https://files.rcsb.org/ligands/download/{}.cif".format(self.aa_name), cif_path)
        parser = cif.read_file(cif_path)
        block = parser.sole_block()
        self.name_to_atom_idx = {}
        self.alt_name_to_atom_idx = {}
        for aid, aaid, atype, aord in zip(block.find_loop("_chem_comp_atom.atom_id"), block.find_loop("_chem_comp_atom.alt_atom_id"), block.find_loop("_chem_comp_atom.type_symbol"), block.find_loop("_chem_comp_atom.pdbx_ordinal")):
            alt_aid = aaid.replace("\"", "")
            aid = aid.replace("\"", "")
            try:
                aord = int(aord)
            except:
                continue

            atom_in_mol = self.mol.GetAtomWithIdx(aord-1)
            if not atom_in_mol.GetSymbol().upper() == atype.upper():
                print("Warning: find inconsistant atoms",
                      f'{atom_in_mol.GetSymbol()} {atype} {self.aa_name}')
            self.name_to_atom_idx[aid] = atom_in_mol.GetIdx()
            self.alt_name_to_atom_idx[alt_aid] = atom_in_mol.GetIdx()

        if len(block.find_loop("_chem_comp_atom.atom_id")) == 0 and len(self.mol.GetAtoms()) == 1:  # ions
            for atom in self.mol.GetAtoms():
                self.name_to_atom_idx[atom.GetSymbol().upper()
                                      ] = atom_in_mol.GetIdx()
                self.alt_name_to_atom_idx[atom.GetSymbol(
                ).upper()] = atom_in_mol.GetIdx()

    def _parse_structure(self, structure):
        self.positions = {}
        use_alt = False
        self.exist_atom_names = None
        if self.init_exist_atoms is not None:
            non_H = 0
            for atom_name in self.init_exist_atoms:
                if not atom_name.upper().startswith("H") and atom_name not in self.name_to_atom_idx:
                    use_alt = True
                if not atom_name.upper().startswith("H"):
                    non_H += 1
            if non_H > 0:
                self.exist_atom_names = self.init_exist_atoms
        if structure is not None:
            non_H = 0
            for atom in structure:
                atom_name = atom.get_name()
                x, y, z = atom.get_coord()
                self.positions[atom_name] = [x, y, z]
                if not atom_name.upper().startswith("H") and atom_name not in self.name_to_atom_idx:
                    use_alt = True
                if not atom_name.upper().startswith("H"):
                    non_H += 1
            if self.exist_atom_names is None:
                if non_H == 0:
                    self.exist_atom_names = set(self.name_to_atom_idx.keys())
                else:
                    self.exist_atom_names = set(self.positions.keys())
        else:
            if self.exist_atom_names is None:
                self.exist_atom_names = set(self.name_to_atom_idx.keys())
        if use_alt:
            self.name_to_atom_dict = self.alt_name_to_atom_idx
        else:
            self.name_to_atom_dict = self.name_to_atom_idx
        # remove not exist atoms & H
        for n in list(self.name_to_atom_dict.keys()):
            if n not in self.exist_atom_names:
                self.name_to_atom_dict.pop(n)
            else:
                atom = self.mol.GetAtomWithIdx(self.name_to_atom_dict[n])
                if atom.GetSymbol() == "H":
                    self.name_to_atom_dict.pop(n)
                    self.exist_atom_names.remove(n)

        # get default cooh and nh2
        self.default_CO = None
        self.default_N = None
        if "C" in self.name_to_atom_dict:
            self.default_CO = self.name_to_atom_idx["C"]
        if "OXT" in self.name_to_atom_dict and not self.include_default_oh_as_exist_atom and (self.init_exist_atoms is None or "OXT" not in self.init_exist_atoms):
            self.exist_atom_names.remove("OXT")
            self.name_to_atom_dict.pop("OXT")
        if "N" in self.name_to_atom_dict:
            self.default_N = self.name_to_atom_dict["N"]

        self.non_H_atoms = len(self.name_to_atom_dict)
        assert self.non_H_atoms > 0, f"{self.non_H_atoms} {self.aa_name}"
        self.atom_to_name_dict = {v: k for k,
                                  v in self.name_to_atom_dict.items()}

    def _init_default_backbone_atoms(self):
        if "N" in self.exist_atom_names and "CA" in self.exist_atom_names and "C" in self.exist_atom_names:
            self.default_backbone_atoms = [
                [self.name_to_atom_idx["N"], self.name_to_atom_idx["CA"], self.name_to_atom_idx["C"]]]
            return

        aa_pattern = Chem.MolFromSmarts(ud_constants.AABackbondPattern)
        matched_aa_pattern = self.mol.GetSubstructMatches(aa_pattern)

        for p in matched_aa_pattern:
            N, CA, C = p[:3]
            if N in self.atom_to_name_dict and CA in self.atom_to_name_dict and C in self.atom_to_name_dict:
                self.default_backbone_atoms = [[N, CA, C]]
                return

        for idx in self.atom_to_name_dict:
            atom = self.mol.GetAtomWithIdx(idx)
            neighbors = atom.GetNeighbors()
            exist_neighbors = []
            for n in neighbors:
                if n.GetIdx() in self.atom_to_name_dict:
                    exist_neighbors.append(n.GetIdx())
            if len(exist_neighbors) > 1:
                self.default_backbone_atoms = [
                    [exist_neighbors[0], atom.GetIdx(), exist_neighbors[1]]]
                return
        first_atom_idx = list(self.atom_to_name_dict.keys())[0]
        self.default_backbone_atoms = [
            [first_atom_idx, first_atom_idx, first_atom_idx]]

    def _init_torsion_angles(self):
        self._inner_torsion_angles_definition, self._is_180_symmetric, self._rmsds_after_180_rotation, self._atom_idxs_affected_by_torsion_angles, self._default_positions, self._reindex_maps, not_changed = autils.get_rotable_bond(
            self.mol, set(self.atom_to_name_dict.keys()))

    def _init_gt_positions_and_default_positions(self):
        # node feats
        self._has_gt_positions = np.zeros(self.non_H_atoms)
        self._gt_positions = np.zeros((self.non_H_atoms, 3))
        self._out_default_positions = np.zeros((self.non_H_atoms, 3))
        self._node_int_feats = np.zeros((self.non_H_atoms, 2), dtype=int)
        self._node_float_feats = np.zeros(
            (self.non_H_atoms, 4), dtype=float)
        self._residue_index = np.zeros(self.non_H_atoms, dtype=int)+self.index
        self.ret_atom_names = []
        self.ret_atom_types = []

        # backbone feats
        # self._pseudo_backbone_atoms = np.zeros([1, 3], dtype=int)
        # self._pseudo_backbone_exists = np.zeros(1, dtype=int)
        # self._pseudo_backbone = np.zeros([1, 4, 4], dtype=float)

        self.atoms = []
        edges = []
        self.edge_attrs = []

        # init node feats
        self._map_from_inner_idx_to_output_idx = {} #文件中的atom_idx到输出中的atom_idx的映射，即实际使用的inx
        exists_atom_idx = sorted(list(self.atom_to_name_dict.keys()))
        assert len(exists_atom_idx) == self.non_H_atoms
        for i in range(self.non_H_atoms):
            inner_idx = exists_atom_idx[i]
            atom_name = self.atom_to_name_dict[inner_idx]
            self._map_from_inner_idx_to_output_idx[inner_idx] = i
            if atom_name in self.positions:
                self._has_gt_positions[i] = 1
                self._gt_positions[i] = np.array(self.positions[atom_name])
            self._out_default_positions[i] = self._default_positions[inner_idx]

            atom = self.mol.GetAtomWithIdx(inner_idx)
            self.atoms.append(atom)
            self._node_int_feats[i, 0] = atom.GetAtomicNum()
            self._node_int_feats[i, 
                                 1] = ud_constants.CIPCodeDict[self.cip_info[inner_idx]]

            self._node_float_feats[i, 0] = ud_constants.PeriodicTable.GetAtomicWeight(
                atom.GetAtomicNum())
            self._node_float_feats[i, 1] = ud_constants.PeriodicTable.GetRvdw(
                atom.GetAtomicNum())
            self._node_float_feats[i, 2] = ud_constants.PeriodicTable.GetRb0(
                atom.GetAtomicNum())
            self._node_float_feats[i, 3] = ud_constants.PeriodicTable.GetNOuterElecs(
                atom.GetAtomicNum())
            self.ret_atom_names.append(atom_name)
            self.ret_atom_types.append(atom.GetSymbol())

        # init edge feats
        for bond in self.mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if start in self._map_from_inner_idx_to_output_idx and end in self._map_from_inner_idx_to_output_idx:
                start_save_idx = self._map_from_inner_idx_to_output_idx[start]
                end_save_idx = self._map_from_inner_idx_to_output_idx[end]
                edges.append([start_save_idx, end_save_idx])
                edges.append([end_save_idx, start_save_idx])
                # self.edge_attrs.append(ud_constants.MolBondDict[str(bond.GetBondType())])
                self.edge_attrs.extend(
                    [ud_constants.MolBondDict[str(bond.GetBondType())], ud_constants.MolBondDict[str(bond.GetBondType())]])
        self._edges = np.array(edges, dtype=int).reshape(-1, 2)
        self._edge_attrs = np.array(self.edge_attrs, dtype=int).reshape(-1)

        

    def init_from_cif_and_sdf_files(
        self,
        structure: object
    ) -> None:
        # parse mol from sdf
        structure = self._load_from_sdf(structure=structure)

        # load mol from cif, merge cif and sdf
        self._load_from_cif()

        # record chirality info
        self.cip_info = {atom.GetIdx(): atom.GetPropsAsDict()[
            '_CIPCode'] if '_CIPCode' in atom.GetPropsAsDict() else "N" for atom in self.mol.GetAtoms()}

        # parse_structures
        self._parse_structure(structure=structure)

        # init default backbone atoms
        # self._init_default_backbone_atoms()

        # init torsion angles
        self._init_torsion_angles()

    def get_graph_with_gt(
        self
    ) -> dict:
        self._init_gt_positions_and_default_positions()

        ret = {
            "nodes_int_feats": self._node_int_feats,
            "nodes_float_feats": self._node_float_feats,
            "nodes_residue_index": self._residue_index,
            "gt_positions": self._gt_positions,
            "gt_position_exists": self._has_gt_positions,
            "edges": self._edges,
            "edge_attrs": self.edge_attrs,
           
            "atom_names": self.ret_atom_names,
            "atom_types": self.ret_atom_types,
            # "aa_names": [self.aa_name for _ in self.ret_atom_names]

        }
        
        ret.update(self._load_pre_emb())
        
        return ret, self.default_CO, self.default_N
