import argparse
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path
import time

import torch
from torch.nn.utils import clip_grad_norm_

from data.dataset import assert_disjoint
from engine import loader, seed_everything, validate, write_json
from losses import SegmentationLoss
from models import ThyroidSegmentation
from progress import Progress


def main():
    p = argparse.ArgumentParser(description='Train medical-image SEGMENTATION only')
    p.add_argument('--config', default='configs/full.json')
    p.add_argument('--data-root'); p.add_argument('--splits'); p.add_argument('--out')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int); p.add_argument('--size', type=int, help='square canvas/max image side; new configs default to 1024')
    p.add_argument('--base', type=int); p.add_argument('--batch-size', type=int)
    p.add_argument('--workers', type=int); p.add_argument('--seed', type=int)
    p.add_argument('--resume')
    p.add_argument('--upscale', action=argparse.BooleanOptionalAction, default=None,
                   help='allow interpolation above native resolution (default disabled in new configs)')
    p.add_argument('--resize-filter', choices=('bilinear','lanczos'))
    p.add_argument('--log-every', type=int, default=10, help='batch interval for redirected logs')
    p.add_argument('--no-progress', action='store_true')
    p.add_argument('--max-train-batches', type=int); p.add_argument('--max-val-batches', type=int)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ('data_root','splits','out','epochs','size','batch_size','workers','seed','upscale','resize_filter'):
        if getattr(args,key) is not None: config[key] = getattr(args,key)
    if args.base is not None: config['model']['base'] = args.base
    if args.log_every < 1: raise ValueError('--log-every must be positive')
    if any(v is not None and v < 1 for v in (args.max_train_batches,args.max_val_batches)):
        raise ValueError('batch limits must be positive')
    if config['epochs'] < 1 or config['batch_size'] < 1: raise ValueError('invalid epochs/batch size')
    out = Path(config['out']); out.mkdir(parents=True,exist_ok=True)
    if not args.resume and (out/'last.pt').exists():
        raise ValueError('run already exists; use --resume or a fresh --out')
    torch.set_num_threads(config.get('cpu_threads',4))
    seed_everything(config['seed'])
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    train, val = loader(config,'train',True), loader(config,'val')
    assert_disjoint(train.dataset.rows,val.dataset.rows)
    model = ThyroidSegmentation(**config['model']).to(device)
    criterion = SegmentationLoss(**config['loss'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    amp = config.get('amp',True) and device.type=='cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'])
    start, best = 0, -1.
    if args.resume:
        ckpt = torch.load(args.resume,map_location='cpu',weights_only=False)
        for key in ('model','size','data_root','splits','seed','batch_size','loss','epochs'):
            if ckpt['config'][key] != config[key]: raise ValueError(f'resume mismatch: {key}')
        for key,legacy in (('upscale',True),('resize_filter','bilinear')):
            if ckpt['config'].get(key,legacy) != config.get(key,legacy):
                raise ValueError(f'resume preprocessing mismatch: {key}; use the saved config.json')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler']); scaler.load_state_dict(ckpt['scaler'])
        start, best = ckpt['epoch']+1, ckpt['best_dice']
    write_json(out/'config.json',config)
    write_json(out/'environment.json',dict(torch=torch.__version__, device=str(device),
        cuda=torch.version.cuda, parameters=sum(p.numel() for p in model.parameters()),
        smoke_run=bool(args.max_train_batches or args.max_val_batches)))
    print(f'device={device} precision={"FP16 autocast + FP32" if amp else "FP32"} '
          f'input={config["size"]}x{config["size"]} upscale={config.get("upscale",True)} '
          f'resize={config.get("resize_filter","bilinear")} '
          f'parameters={sum(p.numel() for p in model.parameters()):,}',flush=True)
    for epoch in range(start,config['epochs']):
        # Epoch-specific seed allows deterministic sample order/augmentation on resume.
        seed_everything(config['seed']+epoch)
        train.generator.manual_seed(config['seed']+epoch)
        model.train(); total, count = 0., 0
        parts_total = defaultdict(float)
        dice_sum, iou_sum = 0., 0.
        epoch_lr = optimizer.param_groups[0]['lr']
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        batches = min(len(train),args.max_train_batches) if args.max_train_batches else len(train)
        with Progress(batches, f'Train {epoch+1}/{config["epochs"]}', args.log_every,
                      enabled=not args.no_progress) as bar:
            for batch in islice(train,batches):
                x,y,v = (batch[k].to(device) for k in ('image','mask','valid'))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=amp):
                    prediction = model(x)
                    loss,parts = criterion(prediction,y,v)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update()
                total += float(loss.detach())*len(x); count += len(x)
                for key,value in parts.items(): parts_total[key] += float(value)*len(x)
                with torch.no_grad():
                    pred_mask = (prediction['segmentation'].float().sigmoid() >= .5)*v
                    truth = y*v
                    intersection = (pred_mask*truth).sum((1,2,3))
                    area = pred_mask.sum((1,2,3))+truth.sum((1,2,3))
                    union = area-intersection
                    dice_sum += float(torch.where(area>0, 2*intersection/area.clamp_min(1), 1.).sum())
                    iou_sum += float(torch.where(union>0, intersection/union.clamp_min(1), 1.).sum())
                fields = dict(loss=f'{total/count:.4f}',
                    **{k:f'{value/count:.4f}' for k,value in parts_total.items()},
                    dice=f'{dice_sum/count:.4f}', iou=f'{iou_sum/count:.4f}', lr=f'{epoch_lr:.2e}',
                    grad=f'{float(grad_norm):.3f}')
                if device.type=='cuda': fields['peak_GPU_GB']=f'{torch.cuda.max_memory_allocated(device)/1024**3:.2f}'
                bar.update(**fields)
        if not count: raise ValueError('no training batches')
        metrics,_ = validate(model,val,device,limit=args.max_val_batches,criterion=criterion,
            description=f'Val {epoch+1}/{config["epochs"]}',progress=not args.no_progress,log_every=args.log_every)
        score = metrics['overall']['dice']
        improved = score>best
        best = max(best,score)
        scheduler.step()
        record = dict(epoch=epoch,train_loss=total/count,
            train_losses={k:v/count for k,v in parts_total.items()},train_dice=dice_sum/count,
            train_iou=iou_sum/count,lr=epoch_lr,val=metrics,seconds=time.perf_counter()-tick)
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
        ckpt = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    scheduler=scheduler.state_dict(),config=config,epoch=epoch,best_dice=best)
        torch.save(ckpt,out/'last.pt')
        if improved: torch.save(ckpt,out/'best.pt')
        print(f'epoch={epoch+1} loss={total/count:.5f} val_dice={score:.5f} best={best:.5f}',flush=True)
    print('Finished. No test-set evaluation was performed during training.')


if __name__=='__main__': main()
