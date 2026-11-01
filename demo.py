"""Run the bundled real-data demo without preparing external features."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import uuid

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["test", "train", "all"], default="test",
                   help="test: pretrained checkpoint; train: one-epoch training then test; all: both")
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    env=os.environ.copy()
    for key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
        env[key]="2"
    tag="demo_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6]
    common=["--config",str(root/"demo/config.json"),"--split","prot_5fold_5","--fold","1"]
    def run(script,args):
        command=[sys.executable,str(root/script),*common,*args]
        print("Running:", " ".join(command),flush=True)
        subprocess.run(command,cwd=root,env=env,check=True)
    if a.mode in ("test", "all"):
        run("test.py", ["--device", a.device, "--output", f"outputs/{tag}/pretrained_test"])
    if a.mode in ("train", "all"):
        run("train.py", ["--device", a.device, "--epochs", "1", "--run-name", tag])
        run("test.py", ["--device", a.device, "--checkpoint",
                        f"outputs/{tag}/prot_5fold_5/fold_1/best.pt",
                        "--output", f"outputs/{tag}/trained_test"])
    print("Demo completed. Outputs:",root/"demo/outputs"/tag)

if __name__=="__main__":
    main()
