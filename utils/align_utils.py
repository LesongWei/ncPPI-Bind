import numpy as np
from rdkit import Chem
from utils.data_constants import SmartsOfRotableBond, SymetricTorsionAngleRmsdThreshold
import rdkit.Chem.rdMolTransforms as rdmt
import networkx as nx


def kabsch_rotation(P, Q):
    C = P.transpose(-1, -2) @ Q
    V, _, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    U = V @ W
    return U


def get_optimal_transform(src_atoms, tgt_atoms):
    src_center = src_atoms.mean(-2)[None, :]
    tgt_center = tgt_atoms.mean(-2)[None, :]
    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x


def compute_rmsd(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
    sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
    msd = np.mean(sd)
    return np.sqrt(msd + eps)


def get_optimal_align(target, ref):
    substract_match = ref.GetSubstructMatches(target, uniquify=False)
    assert len(
        substract_match) > 0, f'{Chem.MolToSmiles(ref)} {Chem.MolToSmiles(target)}'
    for v in target.GetConformers():
        target_id = v.GetId()
    for v in ref.GetConformers():
        ref_id = v.GetId()
    min_rmsds = float("inf")
    ret_map_from_t_to_r = {}
    for s in substract_match:
        target_pos, ref_pos = [], []
        atom_map = {}
        assert len(s) == len(target.GetAtoms())
        for t, r in zip(range(len(s)), s):
            if tuple(target.GetConformers()[target_id].GetAtomPosition(t)) != (-999, -999, -999) and target.GetAtomWithIdx(t).GetSymbol() != "H":
                ref_pos.append(ref.GetConformers()[ref_id].GetAtomPosition(r))
                target_pos.append(target.GetConformers()[
                                  target_id].GetAtomPosition(t))
                assert ref.GetAtomWithIdx(r).GetSymbol(
                ) == target.GetAtomWithIdx(t).GetSymbol()
            atom_map[r] = t

        ref_pos = np.array(ref_pos)
        target_pos = np.array(target_pos)
        if len(ref_pos) == 0:
            min_rmsds = 0
            continue
        optimal_rot, optimal_trans = get_optimal_transform(target_pos, ref_pos)
        rmsd = compute_rmsd(ref_pos, target_pos@optimal_rot+optimal_trans)
        min_rmsds = min(min_rmsds, rmsd)
        if min_rmsds == rmsd:
            ret_map_from_t_to_r = atom_map
    return min_rmsds, ret_map_from_t_to_r


def get_optimal_conformation(target, ref_conforms):
    rmsds_and_atom_map = [get_optimal_align(
        target, ref) for ref in ref_conforms]
    rmsds = [_[0] for _ in rmsds_and_atom_map]
    maps = [_[1] for _ in rmsds_and_atom_map]
    return min(rmsds), maps[np.argmin(rmsds)], ref_conforms[np.argmin(rmsds)]


def mol_to_graph(mol):
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        graph.add_node(atom_idx, symbol=atom_symbol)

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = str(bond.GetBondType())
        graph.add_edge(begin_idx, end_idx,
                       bond_type=bond_type)
    return graph


def get_rotable_bond(mol, exist_atoms: set):
    mol = Chem.Mol(mol)
    rotablePattern = Chem.MolFromSmarts(SmartsOfRotableBond)
    rotableBonds = mol.GetSubstructMatches(rotablePattern)
    ret = []
    is_180_symmetric = []
    rmsds_after_180_rotation = []
    atom_idxs_affected_by_torsion_angle = []
    maps_from_mol_to_new = []

    for v in mol.GetConformers():
        conf = v
    un_changed = set([_ for _ in exist_atoms])
    for bond in rotableBonds:
        begin_atom_idx, end_atom_idx = sorted(bond)
        if not (begin_atom_idx in exist_atoms and end_atom_idx in exist_atoms):
            continue
        neighbors1s = [atom.GetIdx() for atom in mol.GetAtomWithIdx(
            begin_atom_idx).GetNeighbors() if atom.GetIdx() != end_atom_idx and atom.GetIdx() in exist_atoms]
        neighbors2s = [atom.GetIdx() for atom in mol.GetAtomWithIdx(
            end_atom_idx).GetNeighbors() if atom.GetIdx() != begin_atom_idx and atom.GetIdx() in exist_atoms]
        if len(neighbors1s) > 0 and len(neighbors2s) > 0:
            neighbors1 = neighbors1s[0]
            neighbors2 = neighbors2s[0]
        else:
            continue
        assert neighbors1 in exist_atoms
        assert begin_atom_idx in exist_atoms
        assert end_atom_idx in exist_atoms
        assert neighbors1 in exist_atoms
        ret.append((neighbors1, begin_atom_idx,  end_atom_idx, neighbors2))

        ori_torsion_angle = rdmt.GetDihedralDeg(conf, *ret[-1])

        bond_conf = Chem.Conformer(conf)
        bond_conf.SetId(1)
        bond_mol = Chem.Mol(mol)
        Chem.SanitizeMol(bond_mol)

        rdmt.SetDihedralDeg(bond_conf, *ret[-1], ori_torsion_angle+180)
        bond_mol.AddConformer(bond_conf)
        # the key of align_map is the index in mol, the value is index of bond_mol
        rmsd_after_rot, align_map = get_optimal_align(bond_mol, mol)
        rmsds_after_180_rotation.append(rmsd_after_rot)
        if rmsd_after_rot < SymetricTorsionAngleRmsdThreshold:
            is_180_symmetric.append(1)
        else:
            is_180_symmetric.append(0)
        emol = Chem.EditableMol(mol)
        emol.RemoveBond(begin_atom_idx, end_atom_idx)
        graph = mol_to_graph(emol.GetMol())
        connected_graph = list(nx.connected_components(graph))
        assert len(
            connected_graph) == 2, f'{Chem.MolToSmiles(emol.GetMol())} {Chem.MolToSmiles(mol)} {rotableBonds}'
        begin_c_component = [_ for _ in connected_graph if begin_atom_idx in _]
        end_c_component = [_ for _ in connected_graph if end_atom_idx in _]
        assert len(begin_c_component) == 1 and len(end_c_component) == 1
        atom_idxs_affected_by_torsion_angle.append(
            [_ for _ in end_c_component[0] if _ != end_atom_idx])
        for v in [_ for _ in end_c_component[0] if _ != end_atom_idx]:
            if v in un_changed:
                un_changed.remove(v)
        maps_from_mol_to_new.append(align_map)
    positions = conf.GetPositions()
    un_changed = list(un_changed)
    assert len(positions) == len(mol.GetAtoms())
    if len(un_changed) == 0 and len(ret) > 0:
        un_changed = list(ret[0][:3])
    return ret, is_180_symmetric, rmsds_after_180_rotation, atom_idxs_affected_by_torsion_angle, np.array(positions), maps_from_mol_to_new, un_changed
