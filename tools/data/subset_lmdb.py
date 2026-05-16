#!/usr/bin/env python3
"""Create a subset LMDB by randomly sampling N entries from a source LMDB.

Usage:
    python tools/data/subset_lmdb.py
    python tools/data/subset_lmdb.py --n 200
    python tools/data/subset_lmdb.py --train_dir filter_train_hard --n 50
"""

import argparse
import random
from pathlib import Path

import lmdb


def copy_lmdb(src_path, dst_path, idx_map):
    env_src = lmdb.open(str(src_path), max_readers=32, readonly=True,
                        lock=False, readahead=False, meminit=False)
    txn_src = env_src.begin(write=False)

    dst_path.mkdir(parents=True, exist_ok=True)
    map_size = 1 * 1024 * 1024 * 1024
    env_dst = lmdb.open(str(dst_path), map_size=map_size)
    txn_dst = env_dst.begin(write=True)

    copied = 0
    for orig_idx in sorted(idx_map.keys()):
        new_idx = idx_map[orig_idx]
        label = txn_src.get(f'label-{orig_idx:09d}'.encode())
        img = txn_src.get(f'image-{orig_idx:09d}'.encode())
        if label is None or img is None:
            print(f'    WARNING: idx={orig_idx} not found in source LMDB')
            continue
        txn_dst.put(f'label-{new_idx:09d}'.encode(), label)
        txn_dst.put(f'image-{new_idx:09d}'.encode(), img)
        copied += 1

    txn_dst.put(b'num-samples', str(copied).encode())
    txn_dst.commit()
    env_dst.close()
    env_src.close()
    return copied


def main():
    parser = argparse.ArgumentParser(
        description='Create subset LMDB by random sampling')
    parser.add_argument('--train_dir', default='filter_train_challenging')
    parser.add_argument('--data_root', default='~/data/STR/openocr')
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--suffix', default='subset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    ds_base = 'Union14M-L-LMDB-Filtered'

    src_lmdb = data_root / ds_base / args.train_dir
    out_name = f'{args.train_dir}_{args.suffix}'
    dst_lmdb = data_root / ds_base / out_name

    print(f'Source LMDB:  {src_lmdb}')
    print(f'Output LMDB:  {dst_lmdb}')
    print(f'N samples:    {args.n}')

    env = lmdb.open(str(src_lmdb), max_readers=32, readonly=True,
                    lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        num_samples = int(txn.get(b'num-samples'))
    env.close()
    print(f'Source has {num_samples:,} samples')

    rng = random.Random(args.seed)
    n = min(args.n, num_samples)
    selected = sorted(rng.sample(range(1, num_samples + 1), n))
    idx_map = {orig: new for new, orig in enumerate(selected, 1)}

    n_copied = copy_lmdb(src_lmdb, dst_lmdb, idx_map)
    print(f'Copied {n_copied} samples to {dst_lmdb}')


if __name__ == '__main__':
    main()
