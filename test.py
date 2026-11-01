"""Test attnModels checkpoints independently of training (reference: ncPPI_test.py)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.config import DEFAULT_CONFIG, SPLITS, load_config, split_file
from scripts.data import SiteDataset, load_pairs
from scripts.engine import build_model, load_checkpoint, evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG))
    p.add_argument("--split",choices=SPLITS,default="prot_5fold_5")
    p.add_argument("--fold",default="all",help="all, or a single fold number")
    p.add_argument("--checkpoint",help="Optional single-fold checkpoint; relative to project root")
    p.add_argument("--subset",choices=("ncaa","all"),default="ncaa")
    p.add_argument("--device",default="auto")
    p.add_argument("--output",default="outputs/test",help="Output directory relative to project root")
    p.add_argument("--limit",type=int,help="SMOKE TEST ONLY: first N complexes per fold")
    a=p.parse_args()
    if a.limit is not None and a.limit<1: p.error("--limit must be positive")
    if Path(a.output).is_absolute(): p.error("--output must be a project-relative path")
    cfg=load_config(a.config)
    root=Path(a.config).expanduser().resolve().parent
    out=(root/a.output).resolve()
    if not out.is_relative_to(root): p.error("--output must remain inside the project")
    if out.exists() and any(out.iterdir()): p.error("Output directory must be new or empty")
    folds=list(range(1,SPLITS[a.split]+1)) if a.fold=="all" else [int(a.fold)]
    if a.checkpoint and len(folds)!=1: p.error("--checkpoint requires a single --fold")
    ids=set(Path(cfg["data"]["ncaa_peptides"]).read_text().split()) if a.subset=="ncaa" else None
    planned=[]
    for fold in folds:
        pairs=load_pairs(split_file(cfg,a.split,fold,"test"))
        if ids is not None: pairs=[x for x in pairs if x[0] in ids]
        if a.limit: pairs=pairs[:a.limit]
        if not pairs: p.error(f"No selected test complexes in fold {fold}")
        cp=(root/a.checkpoint).resolve() if a.checkpoint else Path(cfg["checkpoints"]["directory"])/a.split/f"fold_{fold}_best.pth"
        if not cp.is_file(): p.error(f"Checkpoint missing: {cp}. Only available checkpoints can be evaluated.")
        planned.append((fold,pairs,cp))
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device=="auto" else a.device)
    out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for fold,pairs,cp in planned:
        model=build_model(cfg).to(device)
        ckpt=load_checkpoint(model,cp,device)
        if "model_state_dict" in ckpt and (ckpt.get("split")!=a.split or ckpt.get("fold")!=fold):
            raise ValueError("Checkpoint split/fold does not match evaluation")
        loader=DataLoader(SiteDataset(pairs,cfg),batch_size=1,shuffle=False,num_workers=cfg["training"]["num_workers"])
        with (out/f"fold_{fold}_predictions.csv").open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["peptide_id","protein_id","side","residue_index","label","probability","evaluated"])
            result=evaluate(model,loader,device,cfg,writer)
        result.update(split=a.split,fold=fold,subset=a.subset,smoke_limit=a.limit,
                      threshold=cfg["evaluation"]["threshold"],
                      checkpoint=str(cp.relative_to(root)) if cp.is_relative_to(root) else cp.name,
                      auprc_definition="average_precision",threshold_rule="probability > threshold",
                      checkpoint_smoke_limit=ckpt.get("smoke_limit"))
        (out/f"fold_{fold}_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
        row={"section":a.split,"fold":fold,"complexes":len(pairs)}
        for side,prefix in [("protein","prot"),("peptide","pep")]:
            for key,col in [("AUC","auc"),("AUPRC","prc"),("MCC","mcc"),("F1","f1"),("ACC","acc")]:
                row[f"{prefix}_{col}"]=result[side][key]
        summaries.append(row)
        print(json.dumps(row),flush=True)
        del model
    with (out/"fold_results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0])); writer.writeheader();writer.writerows(summaries)
    summary={}
    for k in summaries[0]:
        if not k.startswith(("prot_","pep_")): continue
        values=[r[k] for r in summaries if r[k] is not None]
        summary[k]={"mean":float(np.mean(values)) if values else None,
                    "sd":float(np.std(values,ddof=1)) if len(values)>1 else None,"defined_folds":len(values)}
    (out/"mean_sd.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Results written to",a.output,flush=True)

if __name__=="__main__":
    main()
