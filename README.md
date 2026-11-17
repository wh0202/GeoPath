# GeoPath

## Environment

```bash
conda create -n geopath python=3.10 -y
conda activate geopath
cd /path/to/GeoPath

python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt openmim
mim install "mmcv==2.1.0"
```

## Training

```bash
export CDFSOD_PATH=/path/to/datas
export MMGDINOB_PATH=/path/to/grounding_dino_swin-b_pretrain_all-f9818a7c.pth
export BERT_DIR=/path/to/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1

python tools/prepare.py cdfsod --dataset ArTaxOr --shot 5 \
    --out work_dirs/artaxor_5shot.py &&
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 \
    tools/train.py work_dirs/artaxor_5shot.py --launcher pytorch \
    --cfg-options "model.language_model.name=${BERT_DIR}"
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py work_dirs/artaxor_5shot.py \
    work_dirs/artaxor_5shot/best_coco_bbox_mAP_*.pth \
    --work-dir work_dirs/artaxor_5shot_eval \
    --cfg-options "model.language_model.name=${BERT_DIR}"
```
