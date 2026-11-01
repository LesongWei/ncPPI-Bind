from utils.amino_acid import AminoAcid
import utils.data_constants as ud_constants
from openfold.np.residue_constants import restype_1to3, restype_3to1
import numpy as np
from openfold.data.mmcif_parsing import MmcifObject
from rdkit import Chem

add_node_num_feats = [
    "edges", "defined_torsion_angle", "pseudo_backbone_atoms", "nodes_reindex_when_alt_torsion_angles"]
# add_angle_num_feats = [
#     "nodes_affected_by_torsion_angles"
# ]


class Chain():
    def __init__(
        self,
        chain_id: str,
        ligand_path: str,
        aa_seq: str = None,
        aa_seq_with_conn: list = None,
        mmcif_seq_res_to_strcture: list = None,
        mmcif_chain_to_seqres: list = None,
        mmcif_structure: object = None,
        mmcif_object: MmcifObject = None

    ) -> None:
        assert aa_seq_with_conn is not None or mmcif_seq_res_to_strcture is not None or aa_seq is not None, "one of aa_seq, aa_seq_with_conn, mmcif_parsed_object_raw_string should be set"
        assert int(aa_seq is not None)+int(aa_seq_with_conn is not None)+int(
            mmcif_seq_res_to_strcture is not None) == 1, "more than one of aa_seq, aa_seq_with_conn, mmcif_parsed_object_raw_string are set"
        self.ligand_path = ligand_path
        self.chain_id = chain_id

        if aa_seq is not None:
            self.parse_fasta_seq(aa_seq)
        elif aa_seq_with_conn is not None:
            self.parse_aa_seq_with_conn(aa_seq_with_conn)
        elif mmcif_seq_res_to_strcture is not None:
            assert mmcif_chain_to_seqres is not None, "mmcif_inter_conn must be set when mmcif file is used"
            assert mmcif_structure is not None, "mmcif_chain_positions must be set when mmcif file is used"
            self.parse_mmcif_parsed_chain(mmcif_seq_res_to_strcture=mmcif_seq_res_to_strcture,
                                          mmcif_chain_to_seqres=mmcif_chain_to_seqres, mmcif_structure=mmcif_structure, mmcif_object=mmcif_object)

    def parse_fasta_seq(
        self,
        seq: str
    ) -> None:
        """_summary_

        Args:
            seq (str): protein sequence read from fasta file. It should be noted that this is only for standard aas in linear structure
        """
        raise NotImplementedError

    def parse_aa_seq_with_conn(
        self,
        aa_seq_with_conn: list
    ) -> None:
        """_summary_

        Args:
            aa_seq_with_conn (list): in the form of [[aa1_name, aa1_idx, [[conn_atom_name_in_aa1,idx_of_conn_aa1,conn_atom_name_in_conn_aa1, conn_type], ...], [atoms in aa1]], ...]
        """
        raise NotImplementedError

    def get_exist_atom_name(self, mmcif_object: MmcifObject, same_seq_chain_ids: set, aa_idx: int, aa_name: str):
        ret = None
        ori_non_H = 0
        for chain_id in same_seq_chain_ids:
            aa = mmcif_object.seqres_to_structure[chain_id][aa_idx]
            if aa.is_missing or aa_name != aa.name:
                continue
            structure = mmcif_object.structure[chain_id][(
                aa.hetflag,  aa.position.residue_number, aa.position.insertion_code)]
            atom_names = [atom.get_name() for atom in structure]
            num_non_H = sum([1 if not _.upper().startswith(
                "H") else 0 for _ in atom_names])
            if num_non_H > ori_non_H:
                ori_non_H = num_non_H
                ret = set(atom_names)
        return ret

    def parse_mmcif_parsed_chain(
            self,
            mmcif_seq_res_to_strcture: list,
            mmcif_chain_to_seqres: list,
            mmcif_structure: object,
            mmcif_object: MmcifObject
    ) -> None:

        assert not hasattr(self, "_amino_acid_seq")
        assert not hasattr(self, "_non_standard_num")
        assert not hasattr(self, "_dna_rna_num")

        self._amino_acid_seq = {}
        self._dna_rna_num = 0
        self._non_standard_num = 0

        num_res = len(mmcif_chain_to_seqres)
        same_seq_chain_ids = set()
        for k in mmcif_object.chain_to_seqres:
            if mmcif_object.chain_to_seqres[k] == mmcif_chain_to_seqres:
                same_seq_chain_ids.add(k)
        for idx in range(num_res):
            aa = mmcif_seq_res_to_strcture[idx]
            if aa.is_missing:
                structure = None
            else:
                structure = mmcif_structure[(
                    aa.hetflag, aa.position.residue_number, aa.position.insertion_code)]
            self._amino_acid_seq[idx] = AminoAcid(
                aa_name=aa.name,
                aa_idx=idx,
                structure=structure,
                ligand_path=self.ligand_path,
                include_default_oh_as_exist_atom=False,
                is_missing=aa.is_missing,
                init_exist_atoms=self.get_exist_atom_name(
                    mmcif_object=mmcif_object, same_seq_chain_ids=same_seq_chain_ids, aa_idx=idx, aa_name=aa.name)
            )
            if aa.name not in restype_3to1:
                self._non_standard_num += 1
            if aa.name in ud_constants.DNARNAList:
                self._dna_rna_num += 1

    def __getitem__(self, idx) -> AminoAcid:
        return self._amino_acid_seq[idx]

    def get_seq(self) -> str:
        seq = [restype_3to1[_.aa_name]
               if _.aa_name in restype_3to1 else "X" for _ in sorted(self._amino_acid_seq.values(), key=lambda x: x.index)]
        return "".join(seq)

    def get_num_non_standard(self) -> int:
        return self._non_standard_num

    def get_graph_with_gt(self, residue_index=None) -> dict:
        if residue_index is None:
            residue_index = sorted(self._amino_acid_seq.keys())
        ret = None
        num_nodes = 0
        # num_torsion_angles = 0

        default_CO_and_N = {}

        for idx in residue_index:
            aa = self[idx]
            assert aa.index == idx, f'{aa.index} {idx}'
            aa_graph, default_CO, default_N = aa.get_graph_with_gt()

            if ret is None:
                ret = aa_graph
            else:
                for k in list(ret.keys()):
                    if k in add_node_num_feats:
                        ret[k] = np.concatenate([
                            ret[k], aa_graph[k]+num_nodes
                        ], 0)
                    elif k == 'edge_attrs':
                        ret['edge_attrs'] = ret['edge_attrs'] + aa_graph['edge_attrs']
                    # elif k in add_angle_num_feats:
                    #     ret[k] = np.concatenate([
                    #         ret[k], aa_graph[k]+num_torsion_angles
                    #     ], 0)
                    else:
                        ret[k] = np.concatenate([ret[k], aa_graph[k]], 0)
            if default_CO is not None and default_N is not None:
                default_CO_and_N[idx] = [
                    default_CO+num_nodes, default_N+num_nodes]
            elif default_N is not None:
                default_CO_and_N[idx] = [
                    None, default_N+num_nodes]

            num_nodes = len(ret["nodes_int_feats"])
            # num_torsion_angles = len(ret["defined_torsion_angle"])
        
        edge_length = []
        pair_gt_exists_mask = ret["gt_position_exists"][...,
                                                        None]*ret["gt_position_exists"][..., None, :]

        pair_gt_distance = np.sqrt(
            ((ret["gt_positions"][..., None, :]-ret["gt_positions"][..., None, :, :])**2).sum(-1))
        atom_is_s = (ret["nodes_int_feats"][..., 0] == 16).astype(float)
        cross_edges = []
        cross_edge_attrs = []
        assert len(
            pair_gt_distance.shape) == 2 and pair_gt_distance.shape == pair_gt_exists_mask.shape
        has_extra_bond = (pair_gt_distance < ud_constants.BondThreshold) & (
            pair_gt_exists_mask > 0) & (pair_gt_distance > 0) & (abs(ret["nodes_residue_index"][..., None]-ret["nodes_residue_index"][..., None, :]) > 0)
        SSbond = (pair_gt_distance >= ud_constants.BondThreshold) & (
            pair_gt_distance < 2.1) & ((atom_is_s[..., None, :]*atom_is_s[..., None]) == 1)

        has_extra_bond_position = np.nonzero(has_extra_bond | SSbond)

        for i in default_CO_and_N.keys():
            if i in default_CO_and_N and i-1 in default_CO_and_N:
                cur_co, cur_n = default_CO_and_N[i]
                prev_co, prev_n = default_CO_and_N[i-1]
                if prev_co is not None and (self._amino_acid_seq[i].is_missing or self._amino_acid_seq[i-1].is_missing):
                    cross_edges.append([prev_co, cur_n])
                    cross_edges.append([cur_n, prev_co])
                    cross_edge_attrs.append(ud_constants.MolBondDict["SINGLE"])
                    cross_edge_attrs.append(ud_constants.MolBondDict["SINGLE"])
                    edge_length.append(0)
                    # edge_length.append(0)
        for p1, p2 in zip(*has_extra_bond_position):
            edge_length.append(pair_gt_distance[p1, p2])
            cross_edges.append([p1, p2])
            cross_edges.append([p2, p1])
            cross_edge_attrs.append(ud_constants.MolBondDict["SINGLE"])
            cross_edge_attrs.append(ud_constants.MolBondDict["SINGLE"])

        
        # unique_bond = []
        # for pair in cross_edges:
        #     # 检查是否存在对称点
        #     reverse_pair = [pair[1], pair[0]]
        #     if reverse_pair not in unique_bond:
        #         unique_bond.append(pair)

        # cross_edge_attrs = cross_edge_attrs[:len(unique_bond)]

        ret["edge_lengths"] = np.concatenate(
            [np.zeros(len(ret["edges"])), np.array(edge_length)], 0
        )
        # ret["edges"] = np.concatenate([
        #     ret["edges"], np.array(unique_bond).reshape(-1, 2)
        # ], 0)
        ret["edges"] = np.concatenate([
            ret["edges"], np.array(cross_edges).reshape(-1, 2)
        ], 0)
        # ret["edge_attrs"] = np.concatenate([
        #     ret["edge_attrs"], np.array(cross_edge_attrs)
        # ], 0)

        ret["edge_attrs"] = ret["edge_attrs"] + cross_edge_attrs

        # ret["chain_names"] = [self.chain_id for _ in ret["aa_names"]]
        ret["nodes_chain_length"] = np.array(
            [len(self._amino_acid_seq) for _ in ret["nodes_int_feats"]])

        return ret
