import hashlib
import io
import json
import math
import os
import random

import cv2
import lmdb
import msgpack
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from openrec.preprocess import create_operators, transform


_ROTATE_TALL_KEYS = ('RotateTall', 'RotateTallCCW')


def resolve_rotate_tall_config(dataset_config, default_direction='ccw'):
    """Resolve RotateRatioDataSetTVResize's built-in tall-image rotation."""
    cfg = dataset_config.get('rotate_tall', None)

    enabled = True
    ratio = 1.5
    direction = default_direction
    if cfg is None:
        pass
    elif isinstance(cfg, bool):
        enabled = cfg
    elif isinstance(cfg, str):
        mode = cfg.lower()
        enabled = mode not in ('none', 'false', 'off', 'no', '0')
        if enabled:
            direction = mode
    elif isinstance(cfg, dict):
        enabled = cfg.get('enabled', True)
        mode = str(cfg.get('direction', direction)).lower()
        if mode in ('none', 'false', 'off', 'no', '0'):
            enabled = False
        else:
            direction = mode
        ratio = float(cfg.get('ratio', ratio))
    else:
        raise TypeError(f'Unsupported rotate_tall config: {cfg!r}')

    if not enabled:
        return {'enabled': False, 'ratio': ratio, 'direction': direction}
    if direction not in ('cw', 'ccw'):
        raise ValueError(
            f"rotate_tall direction must be 'cw', 'ccw', or 'none', got {direction!r}")
    return {'enabled': True, 'ratio': ratio, 'direction': direction}


def strip_rotate_tall_transforms(dataset_config):
    """RotateRatioDataSetTVResize owns rotation; drop legacy transform entries."""
    transforms = dataset_config.get('transforms', [])
    dataset_config['transforms'] = [
        op for op in transforms
        if not (isinstance(op, dict)
                and any(key in op for key in _ROTATE_TALL_KEYS))
    ]


def adjust_rotate_tall_wh(w, h, rotate_tall):
    w = float(w)
    h = float(h)
    if rotate_tall['enabled'] and h > rotate_tall['ratio'] * w:
        return h, w
    return w, h


def adjust_rotate_tall_ratio(wh_ratio, rotate_tall):
    wh_ratio = float(wh_ratio)
    if (rotate_tall['enabled'] and wh_ratio > 0
            and wh_ratio < 1.0 / rotate_tall['ratio']):
        return 1.0 / wh_ratio
    return wh_ratio


def apply_rotate_tall_image(img, rotate_tall):
    if not rotate_tall['enabled']:
        return img
    if isinstance(img, Image.Image):
        w, h = img.size
        if h <= rotate_tall['ratio'] * w:
            return img
        transpose = (Image.ROTATE_90 if rotate_tall['direction'] == 'ccw'
                     else Image.ROTATE_270)
        return img.transpose(transpose)
    arr = np.asarray(img)
    h, w = arr.shape[0], arr.shape[1]
    if h <= rotate_tall['ratio'] * w:
        return img
    return np.rot90(arr, k=1 if rotate_tall['direction'] == 'ccw' else -1).copy()


class RatioDataSetTVResize(Dataset):

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        super(RatioDataSetTVResize, self).__init__()
        self.ds_width = config[mode]['dataset'].get('ds_width', True)
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        max_ratio = loader_config.get('max_ratio', 10)
        min_ratio = loader_config.get('min_ratio', 1)
        data_dir_list = dataset_config['data_dir_list']
        self.padding = dataset_config.get('padding', True)
        self.padding_rand = dataset_config.get('padding_rand', False)
        self.padding_doub = dataset_config.get('padding_doub', False)
        self.do_shuffle = loader_config['shuffle']
        self.seed = epoch
        data_source_num = len(data_dir_list)
        ratio_list = dataset_config.get('ratio_list', 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        assert (
            len(ratio_list) == data_source_num
        ), 'The length of ratio_list should be the same as the file_list.'
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(
            data_dir_list, ratio_list)
        for data_dir in data_dir_list:
            logger.info('Initialize indexs of datasets:%s' % data_dir)
        self.logger = logger
        # Optional per-LMDB wh_ratio cache: avoids scanning 10M wh keys each
        # init. Cache is keyed on num-samples (invalidates if LMDB changes).
        cache_init = dataset_config.get('cache_init', False)
        self.wh_tables = None
        if cache_init:
            self.wh_tables = {
                i: self._load_or_compute_wh_table(info, logger)
                for i, info in self.lmdb_sets.items()
            }
        self.data_idx_order_list = self.dataset_traversal()
        wh_ratio = np.around(np.array(self.get_wh_ratio()))
        self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)

        # Optional Sample Filtering (ISF). Each filter_mask JSON covers one
        # or more datasets. Per-LMDB semantics:
        #   - drop mode (current): JSON value is {'dropped': [...], 'hashes': [...]}.
        #     Multiple drop-mode masks for same LMDB → union of drop sets (AND
        #     over kept sets via De Morgan).
        #   - keep mode (legacy): JSON value is a plain list of kept idxs.
        #     Multiple keep-mode masks for same LMDB → intersect kept sets.
        #   - mixed modes per LMDB are not supported (raises).
        #   - if no mask references a particular LMDB, that LMDB is left
        #     unfiltered.
        # img_hash (drop mode only) anchors each dropped idx to its scoring-time
        # image bytes — full sweep verifies source LMDB hasn't drifted.
        filter_mask_paths = dataset_config.get('filter_mask_paths', [])
        if isinstance(filter_mask_paths, str):
            filter_mask_paths = [filter_mask_paths]
        if filter_mask_paths and mode == 'Train':
            # lmdb_idx -> {'mode': 'drop'|'keep', 'idxs': set(...),
            #              'drop_hashes': {idx: hash}}
            mask_per_lmdb = {}
            for p in filter_mask_paths:
                p = os.path.expanduser(p)
                logger.info(f'Loading filter mask from {p}')
                with open(p, 'r') as f:
                    mask_json = json.load(f)
                meta = mask_json.get('_meta', {}) if isinstance(
                    mask_json.get('_meta'), dict) else {}
                file_mode = meta.get('mode', 'keep')
                for lmdb_idx, info in self.lmdb_sets.items():
                    dirpath = info['dirpath']
                    matched = None
                    for ds_name in mask_json:
                        if ds_name.startswith('_'):
                            continue
                        if dirpath.endswith(ds_name) or ds_name in dirpath:
                            matched = ds_name
                            break
                    if matched is None:
                        continue
                    payload = mask_json[matched]
                    if file_mode == 'drop':
                        # New format: {'dropped': [...], 'hashes': [...]}
                        if not isinstance(payload, dict):
                            raise ValueError(
                                f'{p}: drop-mode mask expects dict payload '
                                f'with "dropped"+"hashes", got {type(payload).__name__}')
                        dropped = [int(x) for x in payload.get('dropped', [])]
                        hashes = list(payload.get('hashes', []))
                        idx_set = set(dropped)
                        hash_map = {idx: h for idx, h in zip(dropped, hashes) if h}
                    else:
                        # Legacy keep format: payload is a flat list (or {'kept': [...]})
                        kept_list = payload
                        if isinstance(kept_list, dict):
                            kept_list = kept_list.get('kept', [])
                        idx_set = set(int(x) for x in kept_list)
                        hash_map = {}
                    existing = mask_per_lmdb.get(lmdb_idx)
                    if existing is None:
                        mask_per_lmdb[lmdb_idx] = {
                            'mode': file_mode, 'idxs': idx_set,
                            'drop_hashes': dict(hash_map),
                        }
                    else:
                        if existing['mode'] != file_mode:
                            raise ValueError(
                                f'Mixed mask modes for LMDB {dirpath}: '
                                f'{existing["mode"]} vs {file_mode}')
                        if file_mode == 'drop':
                            existing['idxs'] |= idx_set                # union of drops
                            existing['drop_hashes'].update(hash_map)
                        else:
                            existing['idxs'] &= idx_set                # intersect of keeps

            if mask_per_lmdb:
                n_before = len(self.data_idx_order_list)
                def _passes(li, fi):
                    m = mask_per_lmdb.get(li)
                    if m is None:
                        return True
                    return (fi not in m['idxs']) if m['mode'] == 'drop' else (fi in m['idxs'])
                mask_arr = np.array([
                    _passes(int(r[0]), int(r[1]))
                    for r in self.data_idx_order_list])
                self.data_idx_order_list = self.data_idx_order_list[mask_arr]
                self.wh_ratio = self.wh_ratio[mask_arr]
                covered = sorted(mask_per_lmdb.keys())
                modes = sorted({m['mode'] for m in mask_per_lmdb.values()})
                logger.info(
                    f'Filter masks: covered LMDBs={covered}, modes={modes}, '
                    f'kept {len(self.data_idx_order_list)} / {n_before} '
                    f'({len(self.data_idx_order_list) / max(n_before, 1) * 100:.2f}%)')

                # Drop-mode hash verification: full sweep over (small) drop set.
                # Honors hash_check {'full','spot','none'} read below from the
                # same dataset_config — but for ISF drops, 'full' and 'spot' on
                # a few thousand entries are both cheap, so default to full.
                _hc = dataset_config.get('hash_check', 'full')
                _hc_n = int(dataset_config.get('hash_check_n', 1000))
                if _hc != 'none':
                    rng = random.Random(1234)
                    n_total_drops = n_total_hashed = 0
                    n_no_hash = 0
                    for li, m in mask_per_lmdb.items():
                        if m['mode'] != 'drop' or not m['drop_hashes']:
                            n_no_hash += len(m['idxs']) if m['mode'] == 'drop' else 0
                            continue
                        idxs_with_hash = sorted(m['drop_hashes'].keys())
                        if _hc == 'spot' and _hc_n < len(idxs_with_hash):
                            idxs_with_hash = rng.sample(idxs_with_hash, _hc_n)
                        src_txn = self.lmdb_sets[li]['txn']
                        dirpath = self.lmdb_sets[li]['dirpath']
                        for idx in idxs_with_hash:
                            img = src_txn.get(f'image-{idx:09d}'.encode())
                            if img is None:
                                raise ValueError(
                                    f'ISF drop idx={idx} in {dirpath}: source image missing')
                            actual = hashlib.md5(bytes(img)).hexdigest()
                            if actual != m['drop_hashes'][idx]:
                                raise ValueError(
                                    f'ISF hash mismatch in {dirpath} idx={idx}: '
                                    f'source LMDB drifted since str_scores extraction.\n'
                                    f'  stored={m["drop_hashes"][idx]} actual={actual}')
                            n_total_hashed += 1
                        n_total_drops += len(m['idxs'])
                    if n_no_hash:
                        logger.info(
                            f'Filter masks: {n_no_hash} drop idxs without img_hash '
                            f'(legacy mask or extract_str_scores pre-hash) — skipped verification')
                    if n_total_drops:
                        logger.info(
                            f'Filter masks: hash_check={_hc} verified '
                            f'{n_total_hashed}/{n_total_drops} drop idxs')

        for i in range(max_ratio + 1):
            logger.info((1 * (self.wh_ratio == i)).sum())
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)

        # Label sources (train-time only). Two mutually-exclusive modes:
        #   1. label_override_paths (ELC/ILC): JSON patches on top of source labels
        #   2. pl_label_dir_list  (EPL):       parallel pl_selected LMDB per data_dir
        # hash_check: 'full' (default, ~free for ELC's small override subset),
        #             'spot' (random hash_check_n entries per dataset — for EPL
        #             where every sample is overridden),
        #             'none' (skip hash check).
        override_cfg = dataset_config.get('label_override_paths', dataset_config.get('label_override_path', None))
        pl_label_cfg = dataset_config.get('pl_label_dir_list', None)
        if override_cfg and pl_label_cfg:
            raise ValueError("Specify only one of label_override_paths / pl_label_dir_list")
        hash_check = dataset_config.get('hash_check', 'full')
        hash_check_n = int(dataset_config.get('hash_check_n', 1000))
        if hash_check not in ('full', 'spot', 'none'):
            raise ValueError(f"hash_check must be 'full'|'spot'|'none', got {hash_check!r}")
        self.label_overrides = {}  # (lmdb_idx, file_idx) -> new_label
        self.pl_label_sets = {}    # lmdb_idx -> {'env','txn','num_samples','dirpath'}
        if override_cfg and mode == 'Train':
            if isinstance(override_cfg, str):
                override_cfg = [override_cfg]
            # Load and merge all override JSONs
            all_overrides = {}
            for p in override_cfg:
                p = os.path.expanduser(p)
                logger.info(f'Loading label overrides from {p}')
                with open(p, 'r') as f:
                    all_overrides.update(json.load(f))
            # Build (lmdb_idx, file_idx) -> new_label mapping and verify
            rng = random.Random(1234)
            for lmdb_idx, lmdb_info in self.lmdb_sets.items():
                dirpath = lmdb_info['dirpath']
                txn = lmdb_info['txn']
                # Match dataset key: try relative paths
                matched_ds = None
                for ds_name in all_overrides:
                    if dirpath.endswith(ds_name) or ds_name in dirpath:
                        matched_ds = ds_name
                        break
                if matched_ds is None:
                    continue
                overrides = all_overrides[matched_ds]

                # Decide which entries get their image hash checked.
                if hash_check == 'none':
                    hash_check_idxs = set()
                elif hash_check == 'full':
                    hash_check_idxs = None  # sentinel: all
                else:  # 'spot'
                    all_idxs = list(overrides.keys())
                    k = min(hash_check_n, len(all_idxs))
                    hash_check_idxs = set(rng.sample(all_idxs, k)) if k else set()

                n_verified = 0
                n_hash_checked = 0
                for idx_str, entry in overrides.items():
                    file_idx = int(idx_str)
                    # Verify orig label
                    label_key = f'label-{file_idx:09d}'.encode()
                    actual_label = txn.get(label_key)
                    if actual_label is None:
                        logger.info(f'  WARNING: override idx={file_idx} not found in {dirpath}')
                        continue
                    actual_label = actual_label.decode('utf-8')
                    if actual_label.replace(' ', '') != entry['orig'].replace(' ', ''):
                        raise ValueError(
                            f'Label override verification failed: {dirpath} idx={file_idx} '
                            f'expected orig="{entry["orig"]}" but got "{actual_label}"')
                    # Verify image hash (subject to hash_check mode)
                    do_hash = entry.get('img_hash') and (
                        hash_check_idxs is None or idx_str in hash_check_idxs)
                    if do_hash:
                        img_key = f'image-{file_idx:09d}'.encode()
                        img_bytes = txn.get(img_key)
                        if img_bytes:
                            actual_hash = hashlib.md5(img_bytes).hexdigest()
                            if actual_hash != entry['img_hash']:
                                raise ValueError(
                                    f'Image hash verification failed: {dirpath} idx={file_idx} '
                                    f'expected={entry["img_hash"]} got={actual_hash}')
                            n_hash_checked += 1
                    self.label_overrides[(lmdb_idx, file_idx)] = entry['replaced']
                    n_verified += 1
                logger.info(
                    f'  Verified {n_verified} label overrides for {matched_ds} '
                    f'(hash_check={hash_check}, hashed={n_hash_checked})')
            logger.info(f'Total label overrides loaded: {len(self.label_overrides)}')

        # Load PL LMDBs (EPL): one per data_dir, parallel to data_dir_list.
        # Image comes from source LMDB; label comes from msgpack 'pred' in PL LMDB.
        if pl_label_cfg and mode == 'Train':
            if len(pl_label_cfg) != data_source_num:
                raise ValueError(
                    f'pl_label_dir_list length {len(pl_label_cfg)} != '
                    f'data_dir_list length {data_source_num}')
            rng = random.Random(1234)
            for lmdb_idx, pl_dir in enumerate(pl_label_cfg):
                pl_dir = os.path.expanduser(pl_dir)
                if not os.path.isfile(os.path.join(pl_dir, 'data.mdb')):
                    raise FileNotFoundError(f'pl_label_dir_list[{lmdb_idx}]: '
                                            f'no data.mdb at {pl_dir}')
                pl_env = lmdb.open(pl_dir, max_readers=32, readonly=True,
                                   lock=False, readahead=False, meminit=False)
                pl_txn = pl_env.begin(write=False)
                n_pl = int(pl_txn.get(b'num-samples') or 0)
                self.pl_label_sets[lmdb_idx] = {
                    'dirpath': pl_dir, 'env': pl_env, 'txn': pl_txn, 'num_samples': n_pl,
                }

                # Hash verification: stored hash in PL LMDB vs MD5(source image).
                src_txn = self.lmdb_sets[lmdb_idx]['txn']
                src_n = self.lmdb_sets[lmdb_idx]['num_samples']
                if hash_check == 'none':
                    idxs_to_check = []
                elif hash_check == 'full':
                    idxs_to_check = list(range(1, src_n + 1))
                else:  # spot
                    k = min(hash_check_n, src_n)
                    idxs_to_check = rng.sample(range(1, src_n + 1), k) if k else []
                n_hashed = 0
                for idx in idxs_to_check:
                    raw = pl_txn.get(f'image-{idx:09d}'.encode())
                    if raw is None:
                        continue
                    v = msgpack.unpackb(raw, raw=False)
                    stored = v.get('hash')
                    if not stored:
                        continue
                    img_bytes = src_txn.get(f'image-{idx:09d}'.encode())
                    if img_bytes is None:
                        raise ValueError(
                            f'PL label exists but source image missing: '
                            f'{self.lmdb_sets[lmdb_idx]["dirpath"]} idx={idx}')
                    actual = hashlib.md5(bytes(img_bytes)).digest()
                    if actual != stored:
                        raise ValueError(
                            f'PL hash mismatch: {self.lmdb_sets[lmdb_idx]["dirpath"]} '
                            f'idx={idx} (source LMDB changed since epl_select ran)')
                    n_hashed += 1
                logger.info(
                    f'  PL LMDB: {pl_dir} (N={n_pl}) '
                    f'hash_check={hash_check} hashed={n_hashed}/{len(idxs_to_check)}')
            logger.info(f'Total PL LMDBs loaded: {len(self.pl_label_sets)}')

        self.need_reset = True in [x < 1 for x in ratio_list]
        self.error = 0
        self.base_shape = dataset_config.get(
            'base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
        self.base_h = dataset_config.get('base_h', 32)
        self.interpolation = T.InterpolationMode.BICUBIC
        transforms = []
        transforms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.transforms = T.Compose(transforms)

    def get_wh_ratio(self):
        wh_ratio = []
        for idx in range(self.data_idx_order_list.shape[0]):
            lmdb_idx, file_idx = self.data_idx_order_list[idx]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            if self.wh_tables is not None:
                wh_ratio.append(
                    self.adjust_wh_ratio(self.wh_tables[lmdb_idx][file_idx]))
                continue
            wh_key = 'wh-%09d'.encode() % file_idx
            wh = self.lmdb_sets[lmdb_idx]['txn'].get(wh_key)
            if wh is None:
                img_key = f'image-{file_idx:09d}'.encode()
                img = self.lmdb_sets[lmdb_idx]['txn'].get(img_key)
                buf = io.BytesIO(img)
                w, h = Image.open(buf).size
            else:
                wh = wh.decode('utf-8')
                w, h = wh.split('_')
            w, h = self.adjust_wh_for_ratio(w, h)
            wh_ratio.append(float(w) / float(h))
        return wh_ratio

    def adjust_wh_for_ratio(self, w, h):
        return float(w), float(h)

    def adjust_wh_ratio(self, wh_ratio):
        return float(wh_ratio)

    def _load_or_compute_wh_table(self, lmdb_info, logger):
        """Per-LMDB wh_ratio cache indexed by file_idx (1..num_samples).

        Stored at <data_dir>/.wh_ratio_cache_n<num_samples>.npz so it
        self-invalidates when the LMDB is regenerated."""
        dirpath = lmdb_info['dirpath']
        num_samples = lmdb_info['num_samples']
        cache_path = os.path.join(dirpath,
                                  f'.wh_ratio_cache_n{num_samples}.npz')
        if os.path.isfile(cache_path):
            try:
                data = np.load(cache_path)
                arr = data['wh_ratio']
                if arr.shape[0] == num_samples + 1:
                    logger.info(f'Loaded wh cache: {cache_path}')
                    return arr
            except Exception as e:
                logger.info(f'wh cache load failed (rebuilding): {e}')
        logger.info(f'Building wh cache for {dirpath} (one-time, '
                    f'{num_samples} samples)...')
        arr = np.zeros(num_samples + 1, dtype=np.float32)
        txn = lmdb_info['txn']
        for file_idx in range(1, num_samples + 1):
            wh = txn.get(b'wh-%09d' % file_idx)
            if wh is None:
                img = txn.get(b'image-%09d' % file_idx)
                if img is None:
                    continue
                w, h = Image.open(io.BytesIO(img)).size
            else:
                w, h = wh.decode('utf-8').split('_')
            arr[file_idx] = float(w) / float(h)
        try:
            np.savez_compressed(cache_path, wh_ratio=arr)
            logger.info(f'Saved wh cache: {cache_path}')
        except OSError as e:
            logger.info(f'wh cache save failed (ignored): {e}')
        return arr

    def load_hierarchical_lmdb_dataset(self, data_dir_list, ratio_list):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, ratio in zip(data_dir_list, ratio_list):
            dirpath = os.path.expanduser(dirpath)
            env = lmdb.open(dirpath,
                            max_readers=32,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            txn = env.begin(write=False)
            num_samples = int(txn.get('num-samples'.encode()))
            lmdb_sets[dataset_idx] = {
                'dirpath': dirpath,
                'env': env,
                'txn': txn,
                'num_samples': num_samples,
                'ratio_num_samples': int(ratio * num_samples)
            }
            dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['ratio_num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['ratio_num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = list(
                random.sample(range(1, self.lmdb_sets[lno]['num_samples'] + 1),
                              self.lmdb_sets[lno]['ratio_num_samples']))
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data."""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def resize_norm_img(self, data, gen_ratio, padding=True):
        img = data['image']
        w, h = img.size
        if self.padding_rand and random.random() < 0.5:
            padding = not padding
        imgW, imgH = self.base_shape[gen_ratio - 1] if gen_ratio <= 4 else [
            self.base_h * gen_ratio, self.base_h
        ]
        use_ratio = imgW // imgH
        if use_ratio >= (w // h) + 2:
            self.error += 1
            return None
        if not padding:
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(
                    math.ceil(imgH * ratio * (random.random() + 0.5)))
                resized_w = min(imgW, resized_w)
        resized_image = F.resize(img, (imgH, resized_w),
                                 interpolation=self.interpolation)
        img = self.transforms(resized_image)
        if resized_w < imgW and padding:
            # img = F.pad(img, [0, 0, imgW-resized_w, 0], fill=0.)
            if self.padding_doub and random.random() < 0.5:
                img = F.pad(img, [0, 0, imgW - resized_w, 0], fill=0.)
            else:
                img = F.pad(img, [imgW - resized_w, 0, 0, 0], fill=0.)
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = img
        data['valid_ratio'] = valid_ratio
        r = float(w) / float(h)
        data['real_ratio'] = max(1, round(r))
        return data

    def get_lmdb_sample_info(self, txn, index, lmdb_idx=None):
        # EPL path: label comes from pl_selected LMDB, image from source.
        if lmdb_idx is not None and lmdb_idx in self.pl_label_sets:
            pl_txn = self.pl_label_sets[lmdb_idx]['txn']
            raw = pl_txn.get('image-%09d'.encode() % index)
            if raw is None:
                return None
            v = msgpack.unpackb(raw, raw=False)
            label = v.get('pred', '')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            if imgbuf is None:
                return None
            return imgbuf, label, label

        # ELC/ILC path: label from source LMDB, optionally patched by override.
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        orig_label = label
        if lmdb_idx is not None:
            override = self.label_overrides.get((lmdb_idx, index))
            if override is not None:
                label = override
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label, orig_label

    def __getitem__(self, properties):
        img_width = properties[0]
        img_height = properties[1]
        idx = properties[2]
        ratio = properties[3]
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]['txn'], file_idx, lmdb_idx=lmdb_idx)
        if sample_info is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        img, label, orig_label = sample_info
        data = {'image': img, 'label': label, 'orig_label': orig_label}
        outs = transform(data, self.ops[:-1])
        if outs is not None:
            outs = self.resize_norm_img(outs, ratio, padding=self.padding)
            if outs is None:
                ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
                ids = random.sample(ratio_ids, 1)
                return self.__getitem__([img_width, img_height, ids[0], ratio])
            outs = transform(outs, self.ops[-1:])
        if outs is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]


class RotateRatioDataSetTVResize(RatioDataSetTVResize):
    """RatioDataSetTVResize with UnionST-style tall rotation before bucketing."""

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        self.rotate_tall = resolve_rotate_tall_config(config[mode]['dataset'])
        strip_rotate_tall_transforms(config[mode]['dataset'])
        super(RotateRatioDataSetTVResize, self).__init__(
            config, mode, logger, seed=seed, epoch=epoch, task=task)

    def adjust_wh_for_ratio(self, w, h):
        return adjust_rotate_tall_wh(w, h, self.rotate_tall)

    def adjust_wh_ratio(self, wh_ratio):
        return adjust_rotate_tall_ratio(wh_ratio, self.rotate_tall)

    def resize_norm_img(self, data, gen_ratio, padding=True):
        data['image'] = apply_rotate_tall_image(data['image'],
                                                self.rotate_tall)
        return super(RotateRatioDataSetTVResize, self).resize_norm_img(
            data, gen_ratio, padding=padding)
