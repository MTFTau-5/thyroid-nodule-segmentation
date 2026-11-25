import argparse
from inference import validate_scales
from collections import defaultdict
from pathlib import Path

import torch
from data.dataset import assert_disjoint, load_manifest
from engine import checkpoint_model, loader, validate, write_json
from evaluation.metrics import patient_bootstrap


def main():
    p = argparse.ArgumentParser(description='Evaluate segmentation at original resolution')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--data-root'); p.add_argument('--splits'); p.add_argument('--out',default='artifacts/evaluation')
    p.add_argument('--device',default='cpu'); p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--tolerance',type=float,default=2.)
    p.add_argument('--save-masks',action='store_true'); p.add_argument('--limit',type=int)
    p.add_argument('--tta-flip',action='store_true',help='average original and horizontal-flip probabilities')
    p.add_argument('--tta-scales',type=float,nargs='+',default=[1.],help='scale factors, e.g. 0.75 1 1.25')
    args = p.parse_args()
    args.tta_scales = validate_scales(args.tta_scales)
    if not 0<=args.threshold<=1 or args.tolerance<0: raise ValueError('invalid threshold/tolerance')
    torch.set_num_threads(4)
    device=torch.device(args.device)
    model,ckpt=checkpoint_model(args.checkpoint,device)
    config=ckpt['config'].copy()
    if args.data_root: config['data_root']=args.data_root
    if args.splits: config['splits']=args.splits
    manifests=[load_manifest(Path(config['splits'])/f'{s}.jsonl') for s in ('train','val','test')]
    assert_disjoint(*manifests)
    metrics,rows=validate(model,loader(config,args.split),device,args.threshold,args.tolerance,
        args.limit,str(Path(args.out)/'masks') if args.save_masks else None,
        tta_scales=args.tta_scales,tta_flip=args.tta_flip)
    metrics['split']=args.split
    metrics['partial_evaluation']=bool(args.limit)
    metrics['patient_dice_ci95']=patient_bootstrap(rows)
    write_json(Path(args.out)/'metrics.json',metrics)
    write_json(Path(args.out)/'per_image.json',rows)
    print(metrics)


if __name__=='__main__': main()
