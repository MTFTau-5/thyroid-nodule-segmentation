"""Export binary/probability masks and overlays; never attribute labels or prose."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from data.dataset import letterbox, restore
from engine import checkpoint_model, write_json
from inference import predict_probability, validate_scales


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',required=True,help='one image or directory of images')
    p.add_argument('--out',default='artifacts/predictions')
    p.add_argument('--device',default='cpu'); p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--tta-flip',action='store_true',help='average original and horizontal-flip probabilities')
    p.add_argument('--tta-scales',type=float,nargs='+',default=[1.],help='scale factors, e.g. 0.75 1 1.25')
    args=p.parse_args()
    args.tta_scales = validate_scales(args.tta_scales)
    if not 0<=args.threshold<=1: raise ValueError('threshold must be in [0,1]')
    torch.set_num_threads(4)
    device=torch.device(args.device)
    model,ckpt=checkpoint_model(args.checkpoint,device)
    source=Path(args.input)
    paths=sorted(p for p in source.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp')) if source.is_dir() else [source]
    if not paths: raise ValueError('no images found')
    root=source if source.is_dir() else source.parent
    output=Path(args.out).resolve()
    if source.is_dir() and (output==source.resolve() or source.resolve() in output.parents):
        raise ValueError('output directory must be outside input tree')
    summary=[]
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as im: image=im.convert('L')
            prepared,_,_,geometry=letterbox(image,size=ckpt['config']['size'],
                upscale=ckpt['config'].get('upscale',True),
                resize_filter=ckpt['config'].get('resize_filter','bilinear'))
            x=torch.from_numpy(np.asarray(prepared,dtype=np.float32).copy()[None,None]/255.).to(device)
            prob=predict_probability(model,x,args.tta_scales,args.tta_flip)[0,0].cpu().numpy()
            prob=restore(prob,geometry)
            mask=prob>=args.threshold
            # Preserve full basename (including extension) to avoid jpg/png stem collisions.
            destination=output/path.relative_to(root)
            destination.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(mask.astype('uint8')*255).save(str(destination)+'.mask.png')
            Image.fromarray((prob*65535).round().astype('uint16')).save(str(destination)+'.prob.png')
            overlay=np.asarray(image.convert('RGB')).copy()
            overlay[mask]=(overlay[mask]*.6+np.array([255,0,0])*.4).astype('uint8')
            Image.fromarray(overlay).save(str(destination)+'.overlay.png')
            summary.append(dict(image=str(path),geometry=geometry,foreground_pixels=int(mask.sum())))
    write_json(output/'prediction_metadata.json',dict(checkpoint=str(Path(args.checkpoint).resolve()),
        threshold=args.threshold,checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        tta={'scales':list(args.tta_scales),'horizontal_flip':args.tta_flip}, outputs=summary))
    print(f'Wrote {len(paths)} image masks at original resolution to {output}')


if __name__=='__main__': main()
