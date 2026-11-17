# Derived from the GeoPath experiment implementation, built on FT-FSOD.
"""Support-calibrated perturbations of decoder denoising reference boxes."""

import json
import math
from pathlib import Path
from statistics import fmean


def support_statistics(annotation_path):
    """Compute population statistics from one COCO training split."""
    path = Path(annotation_path)
    coco = json.loads(path.read_text(encoding='utf-8'))
    images = {image['id']: image for image in coco.get('images', [])
              if image.get('id') is not None and image.get('width', 0) > 0
              and image.get('height', 0) > 0}
    scales, aspects = [], []
    for annotation in coco.get('annotations', []):
        image = images.get(annotation.get('image_id'))
        box = annotation.get('bbox')
        if (annotation.get('iscrowd', 0) or image is None
                or not isinstance(box, list) or len(box) != 4):
            continue
        width, height = box[2:]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                   for x in (width, height)):
            continue
        width /= image['width']
        height /= image['height']
        scales.append(math.log(math.sqrt(width * height)))
        aspects.append(math.log(width / height))
    if not scales:
        raise ValueError(f'No valid non-crowd support boxes in {path}')

    mean_scale, mean_aspect = fmean(scales), fmean(aspects)
    return dict(
        mean_log_scale=mean_scale,
        std_log_scale=math.sqrt(fmean((x - mean_scale)**2 for x in scales)),
        mean_log_aspect=mean_aspect,
        std_log_aspect=math.sqrt(fmean((x - mean_aspect)**2 for x in aspects)),
        num_boxes=len(scales),
        ann_file=str(path.resolve()),
    )


def sample_perturbations(query, num_dn, probability):
    import torch

    shape = (query.shape[0], num_dn)
    # Preserve the draw order used in the original training runs.
    masks = [torch.rand(shape, device=query.device) < probability
             for _ in range(2)]
    signs = [torch.randint(0, 2, shape, device=query.device).mul(2).sub(1)
             for _ in range(2)]
    return tuple(zip(masks, signs))


def perturb_references(references, mask, sign, axis, std, strength,
                       min_size, max_size):
    import torch

    num_dn = mask.shape[1]
    boxes = references[:, :num_dn]
    cx, cy, width, height = boxes.unbind(-1)
    width, height = width.clamp_min(min_size), height.clamp_min(min_size)
    scale = (width * height).sqrt().log()
    aspect = (width / height).log()
    offset = (sign.to(boxes.dtype) * strength) * boxes.new_tensor(std)
    if axis == 'scale':
        scale = scale + offset
    elif axis == 'aspect':
        aspect = aspect + offset
    else:
        raise ValueError(f'Unknown geometry axis: {axis}')

    width = (scale + aspect / 2).exp().clamp(min_size, max_size)
    height = (scale - aspect / 2).exp().clamp(min_size, max_size)
    width = torch.minimum(width, (2 * torch.minimum(cx, 1 - cx)).clamp_min(min_size))
    height = torch.minimum(height, (2 * torch.minimum(cy, 1 - cy)).clamp_min(min_size))
    shifted = torch.stack((cx, cy, width, height), dim=-1)
    return torch.cat((torch.where(mask[..., None], shifted, boxes),
                      references[:, num_dn:]), dim=1)
