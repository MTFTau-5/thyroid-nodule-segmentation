"""Binary overlap/surface metrics in ORIGINAL image pixels, not millimetres."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def segmentation_metrics(prediction, target, tolerance=2.):
    p, y = np.asarray(prediction, bool), np.asarray(target, bool)
    a, b, intersection = int(p.sum()), int(y.sum()), int((p & y).sum())
    dice = 2*intersection/(a+b) if a+b else 1.
    union = a+b-intersection
    iou = intersection/union if union else 1.
    result = {'dice':dice, 'iou':iou, 'target_empty':not bool(b), 'prediction_empty':not bool(a)}
    if not a or not b:
        both_empty = not (a or b)
        result.update(hd95_px=0. if both_empty else None,
                      boundary_f1=1. if both_empty else 0., surface_dice=1. if both_empty else 0.,
                      surface_distance_undefined=not both_empty)
        return result
    edge_p = p & ~binary_erosion(p, border_value=0)
    edge_y = y & ~binary_erosion(y, border_value=0)
    dp = distance_transform_edt(~edge_y)[edge_p]
    dy = distance_transform_edt(~edge_p)[edge_y]
    precision, recall = float((dp<=tolerance).mean()), float((dy<=tolerance).mean())
    result.update(hd95_px=float(np.percentile(np.concatenate([dp,dy]),95)),
        boundary_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        surface_dice=float(((dp<=tolerance).sum()+(dy<=tolerance).sum())/(len(dp)+len(dy))),
        surface_distance_undefined=False)
    return result


def mean_metrics(rows):
    names = ('dice','iou','hd95_px','boundary_f1','surface_dice')
    result = {'samples':len(rows)}
    for name in names:
        values = [r[name] for r in rows if r[name] is not None]
        result[name] = float(np.mean(values)) if values else None
    result['undefined_hd95_count'] = sum(r['hd95_px'] is None for r in rows)
    return result


def patient_bootstrap(rows, metric='dice', seed=42, repeats=1000):
    """Resample patients, preserving all their views. Equal weight per patient."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            groups[row['patient_id']].append(row[metric])
    values = np.array([np.mean(v) for v in groups.values()])
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(repeats)]
    return dict(mean=float(values.mean()), low=float(np.percentile(means,2.5)),
                high=float(np.percentile(means,97.5)), patients=len(values), seed=seed)
