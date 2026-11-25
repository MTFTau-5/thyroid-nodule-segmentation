import argparse
import time
import torch
from models import ThyroidSegmentation

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu'); p.add_argument('--size',type=int,default=512)
    p.add_argument('--base',type=int,default=64);p.add_argument('--rounds',type=int,default=10)
    args=p.parse_args()
    torch.set_num_threads(4)
    device=torch.device(args.device)
    model=ThyroidSegmentation(base=args.base).to(device).eval()
    x=torch.rand(1,1,args.size,args.size,device=device)
    with torch.inference_mode():
        for _ in range(3): model(x)
        if device.type=='cuda': torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(args.rounds): output=model(x)
        if device.type=='cuda': torch.cuda.synchronize()
    print(dict(parameters=sum(p.numel() for p in model.parameters()),
        mean_ms=(time.perf_counter()-start)/args.rounds*1000,
        peak_cuda_mb=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
        output_shapes={k:list(v.shape) for k,v in output.items()}))
