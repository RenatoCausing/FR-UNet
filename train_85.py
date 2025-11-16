import argparse
import json

from train_pipeline import run_training_pipeline

import torch.multiprocessing 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FR-UNet on the base (85 percent) training split and evaluate on the test split.")
    parser.add_argument('-dp', '--dataset-path', required=True,
                        help='Path to the preprocessed ALL dataset (folder containing training_pro/test_pro/etc).')
    parser.add_argument('-c', '--config', default='config.yaml',
                        help='Path to the YAML config file (default: config.yaml).')
    parser.add_argument('-bs', '--batch-size', type=int, default=1024,
                        help='Batch size for training (default: 1024).')
    parser.add_argument('-nw', '--num-workers', type=int, default=-1,
                        help='Number of DataLoader workers (default: auto; set explicitly to override).')
    parser.add_argument('--pin-memory', dest='pin_memory', action='store_true', default=None,
                        help='Force pin_memory on.')
    parser.add_argument('--no-pin-memory', dest='pin_memory', action='store_false',
                        help='Force pin_memory off.')
    parser.add_argument('--cache-ram', dest='cache_ram', action='store_true', default=True,
                        help='Preload all patches into RAM before training (default: enabled).')
    parser.add_argument('--no-cache-ram', dest='cache_ram', action='store_false',
                        help='Disable RAM caching if memory is constrained.')
    parser.add_argument('--val', action='store_true', default=False,
                        help='Split ten percent of training patches for validation (default: disabled).')
    parser.add_argument('--val-split', type=float, default=0.9,
                        help='Fraction of patches reserved for training when --val is enabled (default: 0.9).')
    parser.add_argument('-o', '--output-root', default='runs/train_85',
                        help='Directory where pipeline artifacts will be written (default: runs/train_85).')
    parser.add_argument('--show-test', action='store_true', default=False,
                        help='Forward --show to the tester to save qualitative predictions.')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='Force training onto a specific device (default: auto-detect).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_training_pipeline(
        include_holdout=False,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        with_val=args.val,
        val_split=args.val_split,
        config_path=args.config,
        output_root=args.output_root,
        show_preds=args.show_test,
        device_preference=None if args.device == 'auto' else args.device,
        pin_memory=args.pin_memory,
        cache_dataset=args.cache_ram
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
