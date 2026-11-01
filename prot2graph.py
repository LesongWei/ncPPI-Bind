"""Prepare protein graphs from mmCIF structures.

Adapted from the author's CPEPAF/prot2graph.py. The 10-A graph rule and
missing-coordinate sequential edges are retained for feature compatibility.
"""
import argparse
import json
import pickle
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import numpy as np
import torch
from scripts.config import ROOT, DEFAULT_CONFIG, SPLITS, load_config, split_file
from scripts.data import load_pairs,protein_key

def representative_coord(residue):
    if "CA" in residue: return residue["CA"].get_coord()
    if "CB" in residue: return residue["CB"].get_coord()
    atoms=[a.get_coord() for a in residue if not a.get_name().upper().startswith("H")]
    if not atoms: raise ValueError("Resolved residue has no representative heavy atoms")
    return np.mean(atoms,axis=0)

def build_one(pair, cif_dir, graph_dir, overwrite=False):
    from openfold.data.mmcif_parsing import parse
    try:
        pdb,chain=pair[1].split("_",1)
        target=Path(graph_dir)/(protein_key(pair)+".pkl")
        path=Path(cif_dir)/(pdb+".cif")
        mmcif=parse(file_id=pdb+".cif",mmcif_string=path.read_text()).mmcif_object
        if mmcif is None: raise ValueError("mmCIF parser returned no structure")
        seq=mmcif.chain_to_seqres[chain]
        mapping=mmcif.seqres_to_structure[chain]
        coordinates=np.zeros((len(seq),3),dtype=float)
        valid=np.zeros(len(seq),dtype=bool)
        for i in range(len(seq)):
            aa=mapping[i]
            if aa.is_missing: continue
            valid[i]=True
            residue=mmcif.structure[chain][(aa.hetflag,aa.position.residue_number,aa.position.insertion_code)]
            coordinates[i]=representative_coord(residue)
        xyz=torch.tensor(coordinates)
        distance=((xyz[...,None,:]-xyz[None,:,:])**2).sum(-1).sqrt()
        geometric=torch.nonzero((distance<=10)&(distance>0))
        sequential=[]
        for i in range(len(seq)):
            if not valid[i]:
                if i>0: sequential.extend([[i,i-1],[i-1,i]])
                if i<len(seq)-1: sequential.extend([[i,i+1],[i+1,i]])
        # Preserve source edge ordering and duplicate sequential edges.
        edges=torch.cat([geometric,torch.tensor(sequential,dtype=torch.long).reshape(-1,2)],dim=0)
        lengths=[torch.norm(xyz[s]-xyz[t]) if valid[s] and valid[t] else xyz.new_tensor(10.) for s,t in edges]
        graph={"edge":edges,"edge_attr":torch.stack(lengths) if lengths else xyz.new_empty((0,)),
               "valid_mask":torch.tensor(valid)}
        if not target.exists() or overwrite:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("wb") as f: pickle.dump(graph,f)
        return {"pair":pair,"protein":pair[1],"sequence":seq,"error":None}
    except Exception as e:
        return {"pair":pair,"error":f"{type(e).__name__}: {e}"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG))
    p.add_argument("--split",choices=SPLITS,default="prot_5fold_5")
    p.add_argument("--cif-dir",default="data/mmcif_files")
    p.add_argument("--out-dir",default="features/protein_graph")
    p.add_argument("--sequence-output",default="data/protein_sequences.json")
    p.add_argument("--error-log",default="outputs/protein_graph_errors.json")
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--limit",type=int,help="Build only first N selected pairs")
    p.add_argument("--overwrite",action="store_true")
    a=p.parse_args()
    cfg=load_config(a.config); root=Path(a.config).expanduser().resolve().parent
    def output(value):
        path=Path(value)
        if path.is_absolute() or not (root/path).resolve().is_relative_to(root):
            p.error("Output paths must be project-relative")
        return (root/path).resolve()
    out,seqout,errout=map(output,[a.out_dir,a.sequence_output,a.error_log])
    cif=Path(a.cif_dir).expanduser(); cif=cif if cif.is_absolute() else root/cif
    if a.workers<1 or (a.limit is not None and a.limit<1): p.error("workers and limit must be positive")
    pairs=set()
    for fold in range(1,SPLITS[a.split]+1):
        # All folds are enumerated: a single pair-disjoint fold is incomplete.
        for part in ("train","val","test"): pairs.update(load_pairs(split_file(cfg,a.split,fold,part)))
    pairs=sorted(pairs)
    if a.limit: pairs=pairs[:a.limit]
    worker=partial(build_one,cif_dir=str(cif),graph_dir=str(out),overwrite=a.overwrite)
    if a.workers>1:
        with Pool(a.workers) as pool: results=list(pool.imap(worker,pairs))
    else: results=[worker(pair) for pair in pairs]
    errors=[r for r in results if r["error"]]
    sequences=json.loads(seqout.read_text()) if seqout.exists() else {}
    for r in results:
        if not r["error"]: sequences[r["protein"]]=r["sequence"]
    seqout.parent.mkdir(parents=True,exist_ok=True);seqout.write_text(json.dumps(sequences,indent=2)+"\n")
    errout.parent.mkdir(parents=True,exist_ok=True);errout.write_text(json.dumps(errors,indent=2)+"\n")
    print(json.dumps({"requested":len(pairs),"completed":len(pairs)-len(errors),"failed":len(errors)}))
    if errors: raise SystemExit(1)

if __name__=="__main__": main()
