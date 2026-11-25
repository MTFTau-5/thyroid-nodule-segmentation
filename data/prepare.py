"""Audit PNG pairs; split by patient directory, merging exact image duplicates."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image


def prepare(root, output, seed=42, holdout_centre=None):
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f'{s}.jsonl').exists() for s in ('train', 'val', 'test')):
        raise ValueError('split files already exist; use a new output directory to keep the test set fixed')
    samples, excluded, orphan_masks = [], [], []
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(a)] = find(b)
    hashes = defaultdict(list)
    for centre in sorted(root.glob('centre_*')):
        image_root, mask_root = centre/'images', centre/'masks'
        for image in sorted(image_root.rglob('*.png')):
            mask = mask_root / image.relative_to(image_root)
            if not mask.is_file():
                excluded.append({'image': str(image.relative_to(root)), 'reason': 'missing_mask'})
                continue
            try:
                with Image.open(image) as im:
                    size = im.size
                    # Hash decoded grayscale pixels, not container bytes.
                    content = im.convert('L').tobytes()
                with Image.open(mask) as im:
                    a = np.asarray(im.convert('L'))
                    if im.size != size:
                        raise ValueError('image/mask dimensions differ')
                    values = np.unique(a).tolist()
                    if not set(values) <= {0, 1, 255}:
                        raise ValueError(f'nonbinary mask values: {values[:12]}')
            except Exception as e:
                excluded.append({'image': str(image.relative_to(root)), 'reason': str(e)})
                continue
            patient = f'{centre.name}/{image.parent.name}'
            find(patient)
            row = dict(image=str(image.relative_to(root)), mask=str(mask.relative_to(root)),
                       patient_id=patient, image_id=image.stem, centre=centre.name,
                       width=size[0], height=size[1], foreground=int((a > 0).sum()))
            digest = hashlib.sha256(str(size).encode() + content).hexdigest()
            mask_digest = hashlib.sha256((a > 0).tobytes()).hexdigest()
            hashes[digest].append((patient, mask_digest, row['image']))
            samples.append(row)
        for mask in sorted(mask_root.rglob('*.png')):
            if not (image_root/mask.relative_to(mask_root)).is_file():
                orphan_masks.append(str(mask.relative_to(root)))
    conflicting = []
    for matches in hashes.values():
        for patient, _, _ in matches[1:]:
            union(matches[0][0], patient)
        if len({x[1] for x in matches}) > 1:
            conflicting.append([x[2] for x in matches])
    if conflicting:
        # Conflicting targets should be resolved explicitly rather than silently selected.
        (output/'conflicts.json').write_text(json.dumps(conflicting, indent=2), encoding='utf-8')
        raise ValueError('identical images have conflicting masks; see conflicts.json')
    grouped = defaultdict(list)
    for row in samples:
        row['group_id'] = find(row['patient_id'])
        grouped[row['group_id']].append(row)
    if holdout_centre and holdout_centre not in {r['centre'] for r in samples}:
        raise ValueError('unknown holdout centre')
    strata = defaultdict(list)
    splits = {s: [] for s in ('train', 'val', 'test')}
    for group, rows in grouped.items():
        centres = {r['centre'] for r in rows}
        if holdout_centre in centres:
            splits['test'].extend(rows)
        else:
            strata['+'.join(sorted(centres))].append(group)
    rng = random.Random(seed)
    for groups in strata.values():
        groups.sort()
        rng.shuffle(groups)
        n = len(groups)
        if n < 3:
            raise ValueError('too few patient groups to split safely')
        nv = max(1, round(n * (.2 if holdout_centre else .15)))
        nt = 0 if holdout_centre else max(1, round(n*.15))
        for i, group in enumerate(groups):
            split = 'val' if i < nv else ('test' if i < nv+nt else 'train')
            splits[split].extend(grouped[group])
    if not all(splits.values()):
        raise ValueError('all three splits must be nonempty')
    summary = dict(data_root=str(root), seed=seed, holdout_centre=holdout_centre,
                   patient_id_source='centre + image parent directory; no verified cross-centre patient linkage',
                   excluded=excluded, orphan_masks=orphan_masks,
                   exact_duplicate_groups=sum(len(v)>1 for v in hashes.values()), splits={})
    for split, rows in splits.items():
        rows.sort(key=lambda r:r['image'])
        with (output/f'{split}.jsonl').open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        summary['splits'][split] = dict(samples=len(rows), patients=len({r['patient_id'] for r in rows}),
            groups=len({r['group_id'] for r in rows}), centres=dict(Counter(r['centre'] for r in rows)),
            empty_masks=sum(r['foreground']==0 for r in rows))
    for a, b in (('train','val'),('train','test'),('val','test')):
        assert not ({r['group_id'] for r in splits[a]} & {r['group_id'] for r in splits[b]})
    (output/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='data')
    p.add_argument('--out', default='artifacts/splits')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--holdout-centre', help='e.g. centre_DY; place whole centre and duplicate-linked groups in test')
    args = p.parse_args()
    report = prepare(args.root, args.out, args.seed, args.holdout_centre)
    print(json.dumps({k:report[k] for k in ('splits','exact_duplicate_groups')}, ensure_ascii=False, indent=2))
    print(f'Excluded {len(report["excluded"])} images; see audit.json for details.')
