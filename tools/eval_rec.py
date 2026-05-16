"""Run Trainer.eval on a single LMDB dataset.

Examples
--------
    python tools/eval_rec.py \
        -c configs/rec/vlreader/vit_vlreader_phase2.yml \
        --data_dir ~/data/STR/openocr/evaluation/IIIT5k

    python tools/eval_rec.py \
        -c configs/rec/vlreader/vit_vlreader_phase2.yml \
        -o Global.pretrained_model=output/rec/vlreader/best.pth
"""

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Eval LMDB dir; overrides Eval.dataset.{data_dir,data_dir_list}.')
    parser.add_argument(
        '--save_errors',
        action='store_true',
        default=False,
        help='Save error images to <pretrained_dir>/eval_errors/<dataset_name>/')
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        default=False,
        help='Save predictions CSV to '
        '<pretrained_dir>/eval_predictions/<dataset_name>__<tag>.csv')
    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        help='Suffix for the predictions filename. '
        'Default: <decoder_name>.')
    parser.add_argument(
        '--no_gt',
        action='store_true',
        default=False,
        help='Run inference on unlabeled data (skip metrics).')
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Print per-key metrics.')
    args = parser.parse_args()
    return args


def main():
    flags = parse_args()
    assert flags.config is not None, 'Please specify -c <config_path>.'
    cfg = Config(flags.config)
    cfg.merge_dict(flags.opt)

    eval_ds = cfg.cfg['Eval']['dataset']
    if flags.data_dir is not None:
        path = os.path.expanduser(flags.data_dir)
        if 'RatioDataSet' in eval_ds['name']:
            eval_ds['data_dir_list'] = [path]
        else:
            eval_ds['data_dir'] = path
    if flags.no_gt:
        eval_ds['no_gt'] = True

    output_dir = cfg.cfg['Global'].get('output_dir')
    if output_dir and output_dir.endswith('/'):
        cfg.cfg['Global']['output_dir'] = output_dir[:-1]
    if cfg.cfg['Global'].get('pretrained_model') is None:
        cfg.cfg['Global']['pretrained_model'] = (
            cfg.cfg['Global']['output_dir'] + '/best.pth')

    cfg.cfg['Global']['use_amp'] = False
    cfg.cfg['PostProcess']['with_ratio'] = True
    cfg.cfg['Metric']['with_ratio'] = True
    cfg.cfg['Metric']['max_len'] = 25
    cfg.cfg['Metric']['max_ratio'] = 12
    keep_keys = eval_ds['transforms'][-1]['KeepKeys']['keep_keys']
    if 'real_ratio' not in keep_keys:
        keep_keys.append('real_ratio')
    if 'file_idx' not in keep_keys:
        keep_keys.append('file_idx')
    if not eval_ds['name'].endswith('Test'):
        eval_ds['name'] = eval_ds['name'] + 'Test'

    trainer = Trainer(cfg, mode='eval')

    if flags.verbose:
        trainer.logger.info('metric in ckpt ***************')
        for k, v in trainer.status.get('metrics', {}).items():
            trainer.logger.info(f'{k}: {v}')

    ds_path = (eval_ds['data_dir_list'][0]
               if 'RatioDataSet' in eval_ds['name']
               else eval_ds['data_dir'])
    ds_name = os.path.expanduser(ds_path).rstrip('/').split('/')[-1]
    pretrained_dir = os.path.dirname(cfg.cfg['Global']['pretrained_model'])

    error_save_dir = None
    if flags.save_errors:
        error_save_dir = os.path.join(pretrained_dir, 'eval_errors', ds_name)
        os.makedirs(error_save_dir, exist_ok=True)
    predictions_save_path = None
    if flags.save_predictions:
        tag = flags.tag or cfg.cfg['Architecture']['Decoder'].get(
            'name', 'pred')
        predictions_save_path = os.path.join(
            pretrained_dir, 'eval_predictions', f'{ds_name}__{tag}.csv')

    metric = trainer.eval(error_save_dir=error_save_dir,
                          dataset_name=ds_name,
                          predictions_save_path=predictions_save_path,
                          no_gt=flags.no_gt)

    if flags.no_gt:
        trainer.logger.info('metric eval skipped (no_gt mode)')
    else:
        trainer.logger.info('metric eval ***************')
        for k, v in metric.items():
            trainer.logger.info(f'{k}: {v}')


if __name__ == '__main__':
    main()
