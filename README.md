# GeoPath

Core modules of GeoPath for cross-domain few-shot object detection.

| File | Module |
| --- | --- |
| `geopath/decoder.py` | Three-path reorganization of six pretrained decoder layers |
| `geopath/geometry.py` | Support-box statistics and Support-Calibrated Geometry Disturbance (SCGD) |
| `geopath/slqr.py` | Localization-gap-conditioned Stochastic Local Query Regularization (SLQR) |
| `geopath/model.py` | Detector/head integration, index-wise prediction aggregation, and progress hook |

The modules depend on PyTorch, MMEngine, and a compatible MMDetection Grounding DINO implementation. Import `geopath.model` to register `GeoPath`, `GeoPathHead`, and `GeoPathProgressHook` with their framework registries.

SLQR defaults to `adaptive_local`: localization gaps determine direction-sampling probabilities, and valid pair losses are averaged uniformly. `GeoPathProgressHook` must be attached to the training runner to update iteration-dependent sampling seeds and loss warm-up. Support geometry statistics must be computed from the training support annotations.

This repository contains the core modules only. Training/testing scripts, experiment configurations, datasets, pretrained weights, and vendored framework code are not included.

The implementation builds on OpenMMLab MMDetection and the FT-FSOD implementation. Original source attribution is retained in the files; the existing Apache-2.0 license is included in `LICENSE`.
