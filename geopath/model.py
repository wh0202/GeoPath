# Adapted from OpenMMLab MMDetection and the FT-FSOD implementation.
"""Grounding DINO with GeoPath decoding, aggregation and regularization."""

import math

import torch
from torch import nn

from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS
from mmengine.structures import InstanceData
from mmdet.models.dense_heads.atss_vlfusion_head import (
    convert_grounding_to_cls_scores,
)
from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.models.layers import SinePositionalEncoding
from mmdet.models.layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerEncoder,
)
from mmdet.registry import MODELS
from mmdet.structures.bbox import (
    bbox_cxcywh_to_xyxy, bbox_overlaps, bbox_xyxy_to_cxcywh,
)

from .decoder import ENDPOINT_INDICES, GeoPathDecoder
from .geometry import sample_perturbations
from .slqr import SLQR_MODES, effective_weight, local_pair_loss


@MODELS.register_module()
class GeoPath(GroundingDINO):
    def __init__(self, *args, support_std, perturb_probability=0.5,
                 perturb_strength=0.5, std_floor=1e-4, min_box_size=1e-4,
                 max_box_size=0.95, **kwargs):
        if len(support_std) != 2 or not all(
                math.isfinite(x) and x >= 0 for x in support_std):
            raise ValueError('Provide two finite support standard deviations.')
        self.support_std = tuple(max(std_floor, x) for x in support_std)
        self.perturb_probability = perturb_probability
        self.geometry_options = dict(strength=perturb_strength,
                                     min_size=min_box_size, max_size=max_box_size)
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        # Keep Grounding DINO module names and initialization order so
        # pretrained and fine-tuned checkpoint keys map to the original layers.
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GeoPathDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        if self.positional_encoding.num_feats * 2 != self.embed_dims:
            raise ValueError('Encoder and positional encoding dimensions disagree.')
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        self.language_model = MODELS.build(self.language_model_cfg)
        # The original Grounding DINO-T checkpoint includes BERT's pooler.
        # Detection uses token features only, so keep its weights for loading
        # but exclude these unused parameters from DDP gradient reduction.
        pooler = self.language_model.language_backbone.body.model.pooler
        if pooler is not None:
            pooler.requires_grad_(False)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims, bias=True)

    def pre_decoder(self, *args, **kwargs):
        decoder_inputs, head_inputs = super().pre_decoder(*args, **kwargs)
        if self.training:
            num_dn = head_inputs['dn_meta']['num_denoising_queries']
            if num_dn:
                samples = sample_perturbations(
                    decoder_inputs['query'], num_dn, self.perturb_probability)
                decoder_inputs['geometry'] = (
                    samples, self.support_std, self.geometry_options)
        return decoder_inputs, head_inputs


@MODELS.register_module()
class GeoPathHead(GroundingDINOHead):
    def __init__(self, *args, slqr_seed=3407, slqr_weight=0.05,
                 slqr_warmup_ratio=0.20, slqr_mode='adaptive_local', **kwargs):
        # The core-only release defaults directly to the paper's adaptive
        # loss, without depending on an external preparation script.
        if slqr_mode not in SLQR_MODES:
            raise ValueError(f'Unknown SLQR mode: {slqr_mode}')
        super().__init__(*args, **kwargs)
        self.slqr_seed = slqr_seed
        self.slqr_weight = slqr_weight
        self.slqr_warmup_ratio = slqr_warmup_ratio
        self.slqr_mode = slqr_mode
        self.slqr_iteration = 0
        self.slqr_total_iterations = 0

    def loss_by_feat(self, all_layers_cls_scores, all_layers_bbox_preds,
                     enc_cls_scores, enc_bbox_preds, batch_gt_instances,
                     batch_img_metas, dn_meta, batch_gt_instances_ignore=None):
        losses = super().loss_by_feat(
            all_layers_cls_scores, all_layers_bbox_preds, enc_cls_scores,
            enc_bbox_preds, batch_gt_instances, batch_img_metas, dn_meta,
            batch_gt_instances_ignore)
        scores, boxes, _, _ = self.split_outputs(
            all_layers_cls_scores, all_layers_bbox_preds, dn_meta)
        scores = scores[list(ENDPOINT_INDICES)]
        boxes = boxes[list(ENDPOINT_INDICES)].mean(dim=0)
        # Padded token logits are -inf. Mask them out for an empty-batch zero.
        total = torch.where(torch.isfinite(scores), scores, 0).sum() * 0
        weight_sum = total.detach()
        seed = self.slqr_seed + 1000003 * self.slqr_iteration + 1000000007 * get_rank()
        for index, (gt, meta) in enumerate(zip(batch_gt_instances, batch_img_metas)):
            if len(gt) == 0:
                continue
            logits = scores[:, index].sigmoid()
            probabilities = []
            for positive_map in gt.positive_maps:
                tokens = (positive_map > 0).nonzero(as_tuple=True)[0].tolist()
                if not tokens:
                    raise ValueError('A ground-truth category has no positive token.')
                per_path = convert_grounding_to_cls_scores(
                    logits, [{1: tokens}] * len(ENDPOINT_INDICES))[..., 0]
                probabilities.append(per_path.mean(dim=0))
            probabilities = torch.stack(probabilities, dim=1)

            with torch.no_grad():
                height, width = meta['img_shape']
                factor = boxes.new_tensor([width, height, width, height])
                gt_boxes = bbox_xyxy_to_cxcywh(gt.bboxes / factor)
                ious = bbox_overlaps(bbox_cxcywh_to_xyxy(boxes[index].detach()),
                                     bbox_cxcywh_to_xyxy(gt_boxes))
            numerator, denominator = local_pair_loss(
                probabilities, ious, seed=seed + index, mode=self.slqr_mode)
            total = total + numerator
            weight_sum = weight_sum + denominator
        loss = total / weight_sum.clamp_min(torch.finfo(weight_sum.dtype).eps)
        weight = effective_weight(
            self.slqr_iteration, self.slqr_total_iterations,
            self.slqr_weight, self.slqr_warmup_ratio)
        losses['loss_local'] = loss * weight
        return losses

    def predict_by_feat(self, all_layers_cls_scores, all_layers_bbox_preds,
                        batch_img_metas, batch_token_positive_maps=None,
                        rescale=False):
        boxes = all_layers_bbox_preds[list(ENDPOINT_INDICES)].mean(dim=0)
        results = []
        for index, meta in enumerate(batch_img_metas):
            token_map = (batch_token_positive_maps[index]
                         if batch_token_positive_maps is not None else None)
            path_scores = all_layers_cls_scores[list(ENDPOINT_INDICES), index].sigmoid()
            if token_map is None:
                path_probs = path_scores
            else:
                path_probs = convert_grounding_to_cls_scores(
                    path_scores, [token_map] * len(ENDPOINT_INDICES))
            probabilities = path_probs.mean(dim=0)
            limit = self.test_cfg.get('max_per_img', len(probabilities))
            if token_map is None:
                scores, queries = probabilities.max(dim=-1).values.topk(limit)
                labels = torch.zeros_like(queries)
            else:
                scores, indices = probabilities.flatten().topk(limit)
                num_classes = probabilities.shape[-1]
                queries, labels = indices // num_classes, indices % num_classes

            detections = bbox_cxcywh_to_xyxy(boxes[index, queries])
            height, width = meta['img_shape']
            detections[:, 0::2] = (detections[:, 0::2] * width).clamp(0, width)
            detections[:, 1::2] = (detections[:, 1::2] * height).clamp(0, height)
            if rescale:
                detections /= detections.new_tensor(meta['scale_factor']).repeat(1, 2)
            results.append(InstanceData(bboxes=detections, scores=scores, labels=labels))
        return results


@HOOKS.register_module()
class GeoPathProgressHook(Hook):
    priority = 'VERY_HIGH'

    def before_train(self, runner):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        runner.logger.info(f'GeoPath SLQR mode: {model.bbox_head.slqr_mode}')

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        model.bbox_head.slqr_iteration = runner.iter + 1
        model.bbox_head.slqr_total_iterations = runner.max_iters
