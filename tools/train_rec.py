import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config, maybe_apply_unionst_transforms
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--eval',
        action='store_true',
        default=True,
        help='Whether to perform evaluation in train',
    )
    parser.add_argument(
        '--restore', '--resume',
        action='store_true',
        default=False,
        help='Resume training from latest.pth in output_dir',
    )
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    restore = FLAGS.restore
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    FLAGS.pop('restore')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    maybe_apply_unionst_transforms(cfg.cfg)

    if restore:
        # Trainer.__init__의 tag 적용 규칙을 그대로 복제: output_dir + tag
        output_dir = cfg.cfg['Global'].get('output_dir', 'output').rstrip('/')
        tag = cfg.cfg['Global'].get('tag', '')
        if tag:
            output_dir = output_dir + tag
        latest = os.path.join(output_dir, 'latest.pth')
        assert os.path.exists(latest), f'latest.pth not found in {output_dir}'
        cfg.cfg['Global']['checkpoints'] = latest
        cfg.cfg['Global']['resume_scheduler'] = True
        cfg.cfg['Global']['wandb_resume'] = True
        cfg.cfg['Global'].setdefault('wandb_rewind', False)
    trainer = Trainer(cfg,
                      mode='train_eval' if FLAGS['eval'] else 'train',
                      task='rec')
    trainer.train()


if __name__ == '__main__':
    main()
