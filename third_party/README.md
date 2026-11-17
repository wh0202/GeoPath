# Third-party code

`mmdet/` is the MMDetection 3.3.0 runtime from the FT-FSOD project snapshot used
in the GeoPath experiments. It retains OpenMMLab copyright headers and includes
the original runtime changes needed by the benchmark transforms and fine-tuning
hooks. GeoPath packaging removes the legacy parallel decoder experiment
registrations and keeps the standard MM Grounding DINO implementation.

The benchmark configurations under `../protocols/` and the MMDetection
training and test entry points under `../tools/` also originate from the
FT-FSOD snapshot. The entry points were modified to use the bundled runtime.

- [OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection)
- [FT-FSOD by Xuanlong Yu et al.](https://github.com/Intellindust-AI-Lab/FT-FSOD)
- [Apache License 2.0](LICENSE)
