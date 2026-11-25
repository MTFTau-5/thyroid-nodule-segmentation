import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset
from .augmentation import UltrasoundAugmentation

BILINEAR = Image.Resampling.BILINEAR
NEAREST = Image.Resampling.NEAREST
RESIZE_FILTERS = {"bilinear": BILINEAR, "lanczos": Image.Resampling.LANCZOS}


def load_manifest(path):
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f'empty manifest: {path}')
    return rows


def assert_disjoint(*manifests):
    groups, patients, images = set(), set(), set()
    for rows in manifests:
        g = {r['group_id'] for r in rows}
        p = {r['patient_id'] for r in rows}
        i = {r['image'] for r in rows}
        if groups & g or patients & p or images & i:
            raise ValueError('patient/group/image overlap across splits')
        groups |= g; patients |= p; images |= i


def letterbox(image, mask=None, size=512, upscale=True, resize_filter="bilinear"):
    w, h = image.size
    if resize_filter not in RESIZE_FILTERS:
        raise ValueError("resize_filter must be bilinear/lanczos")
    scale = min(size/w, size/h, float("inf") if upscale else 1.)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    left, top = (size-nw)//2, (size-nh)//2
    canvas = Image.new('L', (size, size), 0)
    # Do not interpolate native-sized frames. Downsampling uses an antialiasing filter.
    resized = image if (nw,nh)==(w,h) else image.resize((nw,nh), RESIZE_FILTERS[resize_filter])
    canvas.paste(resized, (left,top))
    target = Image.new('L', (size,size), 0)
    if mask is not None:
        target.paste(mask.resize((nw,nh), NEAREST), (left,top))
    valid = np.zeros((size,size), dtype=np.float32)
    valid[top:top+nh,left:left+nw] = 1
    geometry = {'original_width':w, 'original_height':h, 'width':nw, 'height':nh,
                'left':left, 'top':top, 'input_size':size, 'upscale':upscale, 'resize_filter':resize_filter}
    return canvas, target, valid, geometry


def restore(array, geometry, mask=False):
    g = geometry
    crop = array[g['top']:g['top']+g['height'], g['left']:g['left']+g['width']]
    mode = NEAREST if mask else BILINEAR
    return np.asarray(Image.fromarray(crop).resize((g['original_width'],g['original_height']),mode))


class SegmentationDataset(Dataset):
    def __init__(self, root, manifest, size=512, augment=False, upscale=True, resize_filter="bilinear",
                 augmentation=None):
        if size < 32 or size % 32:
            raise ValueError('image size must be >=32 and divisible by 32')
        self.root, self.rows = Path(root), load_manifest(manifest)
        self.size, self.augment = size, augment
        self.upscale, self.resize_filter = upscale, resize_filter
        self.ultrasound_augmentation = (UltrasoundAugmentation(**augmentation)
                                       if augment and augmentation is not None else None)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root/row['image']) as im:
            image = im.convert('L')
        with Image.open(self.root/row['mask']) as im:
            mask = im.convert('L')
        if image.size != mask.size:
            raise ValueError(f'image/mask size mismatch: {row["image"]}')
        if self.augment:
            if random.random() < .5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            # Geometry changes are always shared by image and target.
            angle = random.uniform(-10,10)
            image = image.rotate(angle, resample=BILINEAR)
            mask = mask.rotate(angle, resample=NEAREST)
            image = ImageEnhance.Contrast(image).enhance(random.uniform(.85,1.15))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.9,1.1))
            if self.ultrasound_augmentation is not None:
                image, mask = self.ultrasound_augmentation(image, mask)
        image, mask, valid, geometry = letterbox(image, mask, self.size, self.upscale, self.resize_filter)
        x = np.asarray(image, dtype=np.float32).copy()/255.
        y = (np.asarray(mask)>0).astype(np.float32)
        return dict(image=torch.from_numpy(x[None]), mask=torch.from_numpy(y[None]),
                    valid=torch.from_numpy(valid[None]), index=index,
                    geometry=json.dumps(geometry), image_path=row['image'],
                    patient_id=row['patient_id'], centre=row['centre'])
