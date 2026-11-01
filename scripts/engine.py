"""Training and evaluation shared by the command-line entry points."""
import random
import numpy as np
import torch
import torch.nn.functional as F
from scripts.modelGraph import FullModel
from scripts.ClassCountEMA import ClassCountEMA
from scripts.AdaptiveLambda import AdaptiveLambda
from scripts.CL import MemoryQueue, supcon_with_queue
from scripts.metrics import metrics

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(cfg):
    return FullModel(**cfg["model"])

def load_checkpoint(model, path, device):
    # Checkpoints may contain optimizer/config metadata; load only files you trust.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.removeprefix("module."): v for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    return ckpt

def forward_batch(model, batch, device):
    pa,pe,px,pi,py,ra,re,ry,rx,mask = [x.to(device, non_blocking=True) for x in batch]
    ro,po,rz,pz = model(pa,pe,px,pi,ra,re,rx)
    return (ro.squeeze(0), po.squeeze(0), rz.squeeze(0), pz.squeeze(0), ry.reshape(-1), py.reshape(-1), mask.reshape(-1))

def train_epoch(model, loader, optimizer, device, cfg):
    model.train()
    t = cfg["training"]
    ema = [ClassCountEMA(decay=t["class_ema_decay"]) for _ in range(2)]
    queues = [MemoryQueue(128,t["queue_size"],device) for _ in range(2)]
    adapt = AdaptiveLambda(**t["adaptive"])
    totals = np.zeros(4,dtype=float)
    # EMA, queues and adaptive coefficient state reset each epoch, as in the source.
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        ro,po,rz,pz,ry,py,_ = forward_batch(model,batch,device)
        losses, contrasts = [], []
        for j,(logits,z,y,task) in enumerate(((ro,rz,ry,"protein"),(po,pz,py,"peptide"))):
            mask = y != -1
            y,logits,z = y[mask],logits[mask],z[mask]
            ema[j].update(y)
            # A task with no valid labels contributes zero rather than NaN.
            sup = F.cross_entropy(logits,y,weight=ema[j].weights(task,device=device)) if len(y) else logits.sum()*0
            con = supcon_with_queue(z,y,queues[j],temperature=t["temperature"],pos_only=True) if len(y) else z.sum()*0
            losses.append(sup); contrasts.append(con)
        lp,lq = adapt.step(losses[0].detach(),losses[1].detach(),contrasts[0].detach(),contrasts[1].detach())
        loss = losses[0]+losses[1]+lp*contrasts[0]+lq*contrasts[1]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; check feature/label alignment.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),t["gradient_clip"])
        optimizer.step()
        totals += [float(loss.detach()),float(losses[0].detach()),float(losses[1].detach()),1]
    return totals

@torch.no_grad()
def evaluate(model, loader, device, cfg, writer=None):
    model.eval()
    ema = [ClassCountEMA(decay=cfg["training"]["class_ema_decay"]) for _ in range(2)]
    labels, probs = [[],[]], [[],[]]
    totals = [0.,0.]
    for index,batch in enumerate(loader):
        ro,po,_,_,ry,py,rmask = forward_batch(model,batch,device)
        pair = loader.dataset.pairs[index]  # evaluation loader must be sequential
        for j,(logits,y,task) in enumerate(((ro,ry,"protein"),(po,py,"peptide"))):
            p = F.softmax(logits,dim=-1)[:,1]
            keep = (rmask==1) if j==0 else (y!=-1)
            if writer:
                for k,(label,score) in enumerate(zip(y.cpu().tolist(),p.cpu().tolist())):
                    writer.writerow([pair[0],pair[1],task,k,label,score,int(keep[k])])
            yl,ol,pl = y[keep],logits[keep],p[keep]
            ema[j].update(yl)
            if len(yl):
                totals[j] += F.cross_entropy(ol,yl,weight=ema[j].weights(task,device=device)).item()
            labels[j].extend(yl.cpu().tolist())
            probs[j].extend(pl.cpu().tolist())
    n = len(loader)
    if n==0:
        raise ValueError("No complexes selected for evaluation")
    return {"complexes":n,"loss":sum(totals)/n,
            "protein_loss":totals[0]/n,"peptide_loss":totals[1]/n,
            "protein":metrics(labels[0],probs[0],cfg["evaluation"]["threshold"]),
            "peptide":metrics(labels[1],probs[1],cfg["evaluation"]["threshold"])}
