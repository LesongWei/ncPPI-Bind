"""Train one or all folds. Use python for one device or torchrun for DDP."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from scripts.config import DEFAULT_CONFIG, SPLITS, load_config, split_file
from scripts.data import SiteDataset, load_pairs
from scripts.engine import seed_all, build_model, train_epoch, evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG))
    p.add_argument("--split",choices=SPLITS,default="prot_5fold_5")
    p.add_argument("--fold",default="all",help="all, or a single fold number")
    p.add_argument("--device",default="auto",help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--epochs",type=int)
    p.add_argument("--seed",type=int)
    p.add_argument("--run-name",default="training")
    p.add_argument("--limit",type=int,help="SMOKE TEST ONLY: first N train/val complexes")
    a=p.parse_args()
    if Path(a.run_name).is_absolute() or ".." in Path(a.run_name).parts:
        p.error("--run-name must be project-relative")
    if a.limit is not None and a.limit<1: p.error("--limit must be positive")
    cfg=load_config(a.config)
    if a.epochs is not None: cfg["training"]["epochs"]=a.epochs
    if a.seed is not None: cfg["training"]["seed"]=a.seed
    if cfg["training"]["epochs"]<1: p.error("epochs must be positive")
    world=int(os.environ.get("WORLD_SIZE","1")); rank=int(os.environ.get("RANK","0"))
    local=int(os.environ.get("LOCAL_RANK","0"))
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device=="auto" else a.device)
    if world>1:
        if device.type!="cuda": p.error("DDP training requires CUDA; use python for CPU execution")
        torch.cuda.set_device(local); device=torch.device("cuda",local)
        dist.init_process_group("nccl")
    folds=range(1,SPLITS[a.split]+1) if a.fold=="all" else [int(a.fold)]
    try:
        for fold in folds:
            seed_all(cfg["training"]["seed"])
            tr=load_pairs(split_file(cfg,a.split,fold,"train"))
            va=load_pairs(split_file(cfg,a.split,fold,"val"))
            te=load_pairs(split_file(cfg,a.split,fold,"test"))
            if set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te):
                raise ValueError("Overlapping train/validation/test pairs")
            if not tr or not va: raise ValueError("Empty training/validation set")
            if a.limit: tr,va=tr[:a.limit],va[:a.limit]
            run=Path(cfg["output_dir"])/a.run_name/a.split/f"fold_{fold}"
            status=[None]
            if rank==0:
                if run.exists() and any(run.iterdir()):
                    status[0]=f"Output directory is not empty: {run}. Choose a new --run-name."
                else:
                    run.mkdir(parents=True,exist_ok=True)
                    (run/"config.resolved.json").write_text(json.dumps(cfg,indent=2))
                    (run/"run.json").write_text(json.dumps({"split":a.split,"fold":fold,"world_size":world,"smoke_limit":a.limit,"train_complexes":len(tr),"val_complexes":len(va)},indent=2))
            if world>1: dist.broadcast_object_list(status,src=0)
            if status[0]: raise ValueError(status[0])
            ds=SiteDataset(tr,cfg)
            sampler=DistributedSampler(ds,shuffle=True,seed=cfg["training"]["seed"]) if world>1 else None
            loader=DataLoader(ds,batch_size=1,shuffle=sampler is None,sampler=sampler,
                              num_workers=cfg["training"]["num_workers"],pin_memory=device.type=="cuda")
            vl=DataLoader(SiteDataset(va,cfg),batch_size=1,shuffle=False,num_workers=cfg["training"]["num_workers"]) if rank==0 else None
            model=build_model(cfg).to(device)
            if world>1:
                model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model=DistributedDataParallel(model,device_ids=[local],find_unused_parameters=True)
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg["training"]["learning_rate"])
            schedule=cfg["training"]["scheduler"]
            if schedule not in ("cosine","none"): raise ValueError("scheduler must be cosine or none")
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg["training"]["scheduler_t_max"],eta_min=cfg["training"]["min_learning_rate"]) if schedule=="cosine" else None
            best=float("inf") if cfg["training"]["selection"]=="val_loss" else -float("inf")
            for epoch in range(cfg["training"]["epochs"]):
                if sampler: sampler.set_epoch(epoch)
                lr=optimizer.param_groups[0]["lr"]
                totals=torch.tensor(train_epoch(model,loader,optimizer,device,cfg),device=device)
                if world>1: dist.all_reduce(totals)
                if scheduler: scheduler.step()
                if rank==0:
                    raw=model.module if world>1 else model
                    val=evaluate(raw,vl,device,cfg)
                    score=val["loss"] if cfg["training"]["selection"]=="val_loss" else val["peptide"]["MCC"]
                    improved=score is not None and ((score<best) if cfg["training"]["selection"]=="val_loss" else (score>best))
                    if improved: best=score
                    record={"epoch":epoch+1,"learning_rate":lr,"train_loss":(totals[0]/totals[3]).item(),"validation":val,"selected":improved}
                    with (run/"history.jsonl").open("a") as f: f.write(json.dumps(record)+"\n")
                    ckpt={"model_state_dict":raw.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict() if scheduler else None,"config":cfg,"epoch":epoch+1,"split":a.split,"fold":fold,"selection":cfg["training"]["selection"],"best_score":best,"smoke_limit":a.limit}
                    torch.save(ckpt,run/"last.pt")
                    if improved: torch.save(ckpt,run/"best.pt")
                    print(json.dumps(record),flush=True)
                # Evaluate the entire validation fold once on rank 0. No padded evaluation sampler.
                if world>1: dist.barrier()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()

if __name__=="__main__":
    main()
