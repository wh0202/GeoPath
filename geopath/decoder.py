# Derived from the GeoPath experiment implementation, built on FT-FSOD.
"""Reorganize the six pretrained layers into three independent paths."""

import torch

from mmdet.models.layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder,
)
from mmdet.models.layers.transformer.utils import (
    coordinate_to_encoding, inverse_sigmoid,
)

from .geometry import perturb_references

ENDPOINT_INDICES = (2, 4, 5)


class GeoPathDecoder(GroundingDinoTransformerDecoder):
    def forward(self, query, value, key_padding_mask, self_attn_mask,
                reference_points, spatial_shapes, level_start_index,
                valid_ratios, reg_branches, geometry=None, **kwargs):
        if self.num_layers != 6 or not self.return_intermediate:
            raise ValueError('GeoPath requires six decoder outputs.')
        queries, references = [], []

        def run(index, content, reference):
            references.append(reference)
            reference_input = reference[:, :, None] * torch.cat(
                (valid_ratios, valid_ratios), dim=-1)[:, None]
            position = self.ref_point_head(
                coordinate_to_encoding(reference_input[:, :, 0, :]))
            content = self.layers[index](
                content, query_pos=position, value=value,
                key_padding_mask=key_padding_mask, self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                valid_ratios=valid_ratios, reference_points=reference_input,
                **kwargs)
            queries.append(content)
            return content, (reg_branches[index](content) +
                             inverse_sigmoid(reference, eps=1e-3)).sigmoid()

        shared, refined = run(0, query, reference_points)
        reference = refined.detach()
        scale_reference = aspect_reference = reference
        if geometry is not None:
            samples, std, options = geometry
            (scale_mask, scale_sign), (aspect_mask, aspect_sign) = samples
            scale_reference = perturb_references(
                reference, scale_mask, scale_sign, 'scale', std[0], **options)
            aspect_reference = perturb_references(
                reference, aspect_mask, aspect_sign, 'aspect', std[1], **options)

        scale, scale_box = run(1, shared.clone(), scale_reference)
        run(2, scale, scale_box.detach())
        aspect, aspect_box = run(3, shared.clone(), aspect_reference)
        run(4, aspect, aspect_box.detach())
        run(5, shared, reference)
        # The head refines the input reference of each corresponding layer.
        return torch.stack([self.norm(x) for x in queries]), torch.stack(references)
