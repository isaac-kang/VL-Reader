import hashlib
import io
import math
import os
import random
import re
import unicodedata

import cv2
import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from openrec.preprocess import create_operators, transform
from tools.data.ratio_dataset_tvresize import (adjust_rotate_tall_wh,
                                               apply_rotate_tall_image,
                                               resolve_rotate_tall_config,
                                               strip_rotate_tall_transforms)


class CharsetAdapter:
    """Transforms labels according to the target charset."""

    def __init__(self, target_charset) -> None:
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = re.compile(f'[^{re.escape(target_charset)}]')

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        # Remove unsupported characters
        label = self.unsupported.sub('', label)
        return label


class RatioDataSetTVResizeTest(Dataset):

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        super(RatioDataSetTVResizeTest, self).__init__()
        self.ds_width = config[mode]['dataset'].get('ds_width', True)
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        self.no_gt = dataset_config.get('no_gt', False)
        self.max_file_idx = dataset_config.get('max_file_idx', None)
        loader_config = config[mode]['loader']
        max_ratio = loader_config.get('max_ratio', 10)
        min_ratio = loader_config.get('min_ratio', 1)
        data_dir_list = dataset_config['data_dir_list']
        self.do_shuffle = loader_config['shuffle']
        self.seed = epoch
        self.max_text_length = global_config['max_text_length']
        data_source_num = len(data_dir_list)
        ratio_list = dataset_config.get('ratio_list', 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        assert len(
            ratio_list
        ) == data_source_num, 'The length of ratio_list should be the same as the file_list.'
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(
            data_dir_list, ratio_list)
        for data_dir in data_dir_list:
            logger.info('Initialize indexs of datasets:%s' % data_dir)
        self.logger = logger
        data_idx_order_list = self.dataset_traversal()
        character_dict_path = global_config.get('character_dict_path', None)
        use_space_char = global_config.get('use_space_char', False)
        if character_dict_path is None:
            char_test = '0123456789abcdefghijklmnopqrstuvwxyz'
        else:
            char_test = ''
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip('\n').strip('\r\n')
                    char_test += line
            if use_space_char:
                char_test += ' '
        self.dummy_label = char_test[0] if char_test else 'a'

        # Optional cache for the (slow) bucketing init. Keyed by
        # (data_dirs, ratio_list, char_test, max_text_length, min/max_ratio,
        # no_gt, num_samples) so any param change invalidates it.
        cache_init = dataset_config.get('cache_init', False)
        cache_path = None
        cache_key = None
        if cache_init:
            cache_path, cache_key = self._ratio_cache_path(
                data_dir_list, ratio_list, char_test, min_ratio, max_ratio)
        if cache_path is not None and os.path.isfile(cache_path):
            npz = np.load(cache_path, allow_pickle=True)
            self.data_idx_order_list = npz['data_idx_order_list']
            self.wh_ratio = npz['wh_ratio']
            self.wh_ratio_sort = npz['wh_ratio_sort']
            logger.info(f'Loaded ratio cache: {cache_path} '
                        f'({len(self.wh_ratio)} samples)')
        else:
            wh_ratio, data_idx_order_list = self.get_wh_ratio(
                data_idx_order_list, char_test)
            self.data_idx_order_list = np.array(data_idx_order_list)
            wh_ratio = np.around(np.array(wh_ratio))
            self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)
            self.wh_ratio_sort = np.argsort(self.wh_ratio)
            if cache_path is not None:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    np.savez_compressed(
                        cache_path,
                        data_idx_order_list=self.data_idx_order_list,
                        wh_ratio=self.wh_ratio,
                        wh_ratio_sort=self.wh_ratio_sort,
                        config=np.array(cache_key))
                    logger.info(f'Saved ratio cache: {cache_path}')
                except OSError as e:
                    logger.info(f'Ratio cache save failed (ignored): {e}')
        for i in range(max_ratio + 1):
            logger.info((1 * (self.wh_ratio == i)).sum())
        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)

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

    def get_wh_ratio(self, data_idx_order_list, char_test):
        wh_ratio = []
        wh_ratio_len = [[0 for _ in range(26)] for _ in range(11)]
        data_idx_order_list_filter = []
        charset_adapter = CharsetAdapter(char_test)

        for idx in range(data_idx_order_list.shape[0]):
            lmdb_idx, file_idx = data_idx_order_list[idx]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            if self.max_file_idx is not None and file_idx > self.max_file_idx:
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

            label_key = 'label-%09d'.encode() % file_idx
            label = self.lmdb_sets[lmdb_idx]['txn'].get(label_key)
            if label is not None:
                # return None
                label = label.decode('utf-8')
                # if remove_whitespace:
                label = ''.join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                # if normalize_unicode:
                label = unicodedata.normalize('NFKD',
                                              label).encode('ascii',
                                                            'ignore').decode()
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > self.max_text_length:
                    continue
                label = charset_adapter(label)
                if not label:
                    if not self.no_gt:
                        continue
                    # no_gt: keep sample even if adapter strips everything
                    label = self.dummy_label

                wh_ratio.append(float(w) / float(h))
                wh_ratio_len[int(float(w) /
                                 float(h)) if int(float(w) /
                                                  float(h)) <= 10 else
                             10][len(label) if len(label) <= 25 else 25] += 1
                data_idx_order_list_filter.append([lmdb_idx, file_idx])
            elif self.no_gt:
                # Missing label key: include sample with dummy label
                wh_ratio.append(float(w) / float(h))
                r_bin = int(float(w) / float(h))
                wh_ratio_len[r_bin if r_bin <= 10 else 10][1] += 1
                data_idx_order_list_filter.append([lmdb_idx, file_idx])
        self.logger.info(wh_ratio_len)
        return wh_ratio, data_idx_order_list_filter

    def adjust_wh_for_ratio(self, w, h):
        return float(w), float(h)

    def _ratio_cache_path(self, data_dir_list, ratio_list, char_test,
                          min_ratio, max_ratio):
        """Path for the cached bucketing arrays.

        Stored under <first_data_dir>/.ratio_cache/<hash>.npz. The hash covers
        everything that changes the filter/bucketing outcome so any config
        change invalidates the cache transparently. Returns (path, key_str)
        so the caller can also embed key_str inside the npz for debuggability
        (inspect with `np.load(p, allow_pickle=True)['config']`)."""
        counts = [self.lmdb_sets[i]['num_samples']
                  for i in range(len(self.lmdb_sets))]
        key = repr({
            'data_dir_list': list(data_dir_list),
            'ratio_list': list(ratio_list),
            'char_test': char_test,
            'max_text_length': self.max_text_length,
            'min_ratio': int(min_ratio),
            'max_ratio': int(max_ratio),
            'no_gt': bool(self.no_gt),
            'max_file_idx': (int(self.max_file_idx)
                             if self.max_file_idx is not None else None),
            'num_samples': counts,
            'extra': self._extra_ratio_cache_key(),
        })
        h = hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]
        return os.path.join(data_dir_list[0], '.ratio_cache', f'{h}.npz'), key

    def _extra_ratio_cache_key(self):
        return {}

    def load_hierarchical_lmdb_dataset(self, data_dir_list, ratio_list):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, ratio in zip(data_dir_list, ratio_list):
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
                'ratio_num_samples': int(ratio * num_samples),
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
            img = F.pad(img, [0, 0, imgW - resized_w, 0], fill=0.)
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = img
        data['valid_ratio'] = valid_ratio
        data['gen_ratio'] = imgW // imgH
        r = float(w) / float(h)
        data['real_ratio'] = max(1, round(r))
        return data

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            if not self.no_gt:
                return None
            label = self.dummy_label
        else:
            label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label, label

    def __getitem__(self, properties):
        img_width = properties[0]
        img_height = properties[1]
        idx = properties[2]
        ratio = properties[3]
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        if sample_info is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        img, label, orig_label = sample_info
        data = {'image': img, 'label': label, 'orig_label': orig_label, 'file_idx': file_idx, 'lmdb_idx': lmdb_idx}
        outs = transform(data, self.ops[:-1])
        if outs is not None:
            outs = self.resize_norm_img(outs, ratio, padding=False)
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


class RotateRatioDataSetTVResizeTest(RatioDataSetTVResizeTest):
    """RatioDataSetTVResizeTest with UnionST-style rotation before bucketing."""

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        self.rotate_tall = resolve_rotate_tall_config(config[mode]['dataset'])
        strip_rotate_tall_transforms(config[mode]['dataset'])
        super(RotateRatioDataSetTVResizeTest, self).__init__(
            config, mode, logger, seed=seed, epoch=epoch, task=task)

    def adjust_wh_for_ratio(self, w, h):
        return adjust_rotate_tall_wh(w, h, self.rotate_tall)

    def _extra_ratio_cache_key(self):
        return {'rotate_tall': dict(self.rotate_tall)}

    def resize_norm_img(self, data, gen_ratio, padding=True):
        data['image'] = apply_rotate_tall_image(data['image'],
                                                self.rotate_tall)
        return super(RotateRatioDataSetTVResizeTest, self).resize_norm_img(
            data, gen_ratio, padding=padding)
