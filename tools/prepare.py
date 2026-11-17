"""Build one GeoPath config and compute its training-only geometry statistics."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geopath.geometry import support_statistics


def protocol_path(benchmark, dataset, shot, split_seed):
    if not dataset or Path(dataset).name != dataset or dataset in ('.', '..'):
        raise ValueError('Dataset must be a single protocol name.')
    protocols = ROOT / 'protocols'
    if benchmark == 'cdfsod' and shot in (1, 5, 10) and split_seed is None:
        return (protocols / 'configs_cdfsod/final_configs_bs4' /
                f'grounding_dino_swin-b_finetune_{dataset}_1shot.py')
    if benchmark == 'odinw' and shot in (1, 3, 5, 10) and split_seed in (3, 30, 300):
        return (protocols / 'configs_odinw/final_configs' / f'{shot}shot' /
                f'{shot}shot_seed{split_seed}' /
                f'grounding_dino_swin-l_finetune_{dataset}.py')
    if benchmark == 'rf100' and shot == 10 and split_seed is None:
        return (protocols / 'configs_rf100vlfsod/final_configs' /
                f'grounding_dino_swin-l_finetune_{dataset}_10shot.py')
    raise ValueError('Unsupported shot or split seed for this benchmark.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=('cdfsod', 'odinw', 'rf100'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--shot', type=int, required=True)
    parser.add_argument('--split-seed', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--slqr-seed', type=int,
                        help='Default: 3407, or --seed for CD-FSOD if supplied.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Validate every N epochs (default: 1).')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        source = protocol_path(args.benchmark, args.dataset, args.shot, args.split_seed)
    except ValueError as error:
        parser.error(str(error))
    if not source.is_file():
        parser.error(f'Benchmark protocol not found: {source}')
    if args.val_interval < 1:
        parser.error('--val-interval must be positive')
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error('--seed must be in [0, 2**32)')
    if args.slqr_seed is not None and not 0 <= args.slqr_seed < 2**63:
        parser.error('--slqr-seed must be in [0, 2**63)')

    from mmengine.config import Config

    cfg = Config.fromfile(str(source))
    dataset = cfg.train_dataloader.dataset
    if args.benchmark == 'cdfsod':
        # GeoPath's original 5/10-shot configs inherit the full 1-shot
        # protocol and change only the support annotation file.
        dataset.ann_file = f'annotations/{args.shot}_shot.json'
    annotation = (Path(dataset.data_root) / dataset.ann_file).resolve()
    stats = support_statistics(annotation)

    cfg.model.type = 'GeoPath'
    cfg.model.pop('rand_dnquery_rate', None)
    cfg.model.support_std = [stats['std_log_scale'], stats['std_log_aspect']]
    cfg.model.bbox_head.type = 'GeoPathHead'
    slqr_seed = args.slqr_seed
    if slqr_seed is None:
        slqr_seed = (args.seed if args.benchmark == 'cdfsod' and args.seed is not None
                     else 3407)
    cfg.model.bbox_head.slqr_seed = slqr_seed
    cfg.custom_imports = dict(imports=['geopath.model'], allow_failed_imports=False)
    if args.benchmark in ('cdfsod', 'odinw'):
        cfg.custom_hooks = [dict(type='BBoxHeadFirstHook6',
                                 adjust_scheduler_patience=True,
                                 patience_frozen=3, patience_unfrozen=8)]
    cfg.custom_hooks = list(cfg.get('custom_hooks', [])) + [dict(type='GeoPathProgressHook')]
    cfg.find_unused_parameters = False
    cfg.train_cfg.val_interval = args.val_interval
    if args.seed is not None:
        cfg.randomness.seed = args.seed
    if args.benchmark == 'odinw':
        cfg.default_hooks.checkpoint.update(
            save_last=False, save_optimizer=False, save_param_scheduler=False)
    cfg.geopath_support = stats
    cfg.work_dir = str(args.out.resolve().with_suffix(''))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(args.out))
    print(f'GeoPath config: {args.out.resolve()} ({stats["num_boxes"]} support boxes)')


if __name__ == '__main__':
    main()
