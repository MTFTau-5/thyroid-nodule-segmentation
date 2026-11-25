"""Shared runtime utilities; no clinical-label or language modules."""
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from data.dataset import SegmentationDataset, restore
from evaluation.metrics import segmentation_metrics, mean_metrics, patient_bootstrap
from models import ThyroidSegmentation
from progress import Progress
from inference import predict_probability, validate_scales


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def loader(config, split, training=False):
    dataset = SegmentationDataset(config['data_root'], Path(config['splits'])/f'{split}.jsonl',
                                   config['size'], training, config.get('upscale',True),
                                   config.get('resize_filter','bilinear'), config.get('augmentation'))
    gen = torch.Generator().manual_seed(config['seed'])
    return DataLoader(dataset, batch_size=config['batch_size'] if training else 1,
        shuffle=training, num_workers=config.get('workers',0), pin_memory=False,
        worker_init_fn=seed_worker, generator=gen, persistent_workers=config.get('workers',0)>0)


def checkpoint_model(path, device):
    # Only load checkpoints produced by this code or another trusted source.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = ThyroidSegmentation(**checkpoint['config']['model']).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


@torch.inference_mode()
def validate(model, batches, device, threshold=.5, tolerance=2., limit=None, masks_dir=None,
             criterion=None, description='Validation', progress=True, log_every=10,
             tta_scales=(1.,), tta_flip=False):
    tta_scales = validate_scales(tta_scales)
    model.eval()
    rows, seconds = [], 0.
    losses, seen = defaultdict(float), 0
    dice_sum, iou_sum = 0., 0.
    if masks_dir:
        Path(masks_dir).mkdir(parents=True, exist_ok=True)
    total = min(len(batches),limit) if limit else len(batches)
    if total < 1: raise ValueError('no validation batches')
    with Progress(total, description, every=log_every, enabled=progress) as bar:
        for batch in islice(batches,total):
            x = batch['image'].to(device)
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(x) if criterion is not None else None
            probability = predict_probability(model, x, tta_scales, tta_flip, output)
            if device.type == 'cuda': torch.cuda.synchronize()
            seconds += time.perf_counter()-start
            if criterion is not None:
                loss, parts = criterion(output, batch['mask'].to(device), batch['valid'].to(device))
                losses['loss'] += float(loss)*len(x)
                for key,value in parts.items(): losses[key] += float(value)*len(x)
            seen += len(x)
            for i in range(len(x)):
                geometry = json.loads(batch['geometry'][i])
                prob = restore(probability[i,0].cpu().numpy(), geometry)
                row_index = int(batch['index'][i])
                sample = batches.dataset.rows[row_index]
                with Image.open(batches.dataset.root/sample['mask']) as im:
                    target = np.asarray(im.convert('L'))>0
                prediction = prob >= threshold
                metrics = segmentation_metrics(prediction,target,tolerance)
                metrics.update(image=sample['image'], patient_id=sample['patient_id'], centre=sample['centre'])
                rows.append(metrics)
                dice_sum += metrics['dice']; iou_sum += metrics['iou']
                if masks_dir:
                    path = Path(masks_dir)/sample['image']
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(prediction.astype('uint8')*255).save(path)
            fields = {'dice':f'{dice_sum/len(rows):.4f}', 'iou':f'{iou_sum/len(rows):.4f}'}
            if criterion is not None:
                fields = {**{k:f'{v/seen:.4f}' for k,v in losses.items()}, **fields}
            bar.update(**fields)
    centres = defaultdict(list)
    for row in rows: centres[row['centre']].append(row)
    result = dict(overall=mean_metrics(rows), by_centre={k:mean_metrics(v) for k,v in centres.items()},
        threshold=threshold, surface_tolerance_px=tolerance, distance_unit='original image pixels',
        mean_inference_ms=seconds/max(len(rows),1)*1000,
        tta={'scales':list(tta_scales), 'horizontal_flip':tta_flip})
    if criterion is not None: result['losses'] = {k:v/seen for k,v in losses.items()}
    return result, rows
