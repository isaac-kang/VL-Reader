"""Generate an HTML reproduction-quality report for VL-Reader phase 1.

Runs phase-1 eval on each standard benchmark separately (one LMDB dir per
pass) and writes a single self-contained HTML page with:

  - per-dataset linguistic_acc (top-1 char acc on masked text positions) and
    visual_mse (normalized-pixel MSE on masked patches)
  - a weighted overall row
  - up to 8 image-triplet reconstruction samples per dataset (orig / masked /
    reconstructed) plus the matching gt / masked / recon text strings

The 8-sample cap matches ``Trainer.eval``'s internal ``_phase1_viz_target``;
change it there if you want a denser report.

Usage
-----
    python tools/eval_phase1_report.py \
        -c configs/rec/vlreader/vit_vlreader_phase1.yml \
        -o Global.pretrained_model=output/rec/vlreader/phase1_run1/best.pth

    python tools/eval_phase1_report.py \
        -c configs/rec/vlreader/vit_vlreader_phase1.yml \
        --datasets CUTE80,IIIT5k --output /tmp/p1.html
"""

import base64
import datetime
import html
import io
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from PIL import Image

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


HTML_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 24px; color: #222; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 32px; }
h3 { margin-top: 24px; margin-bottom: 8px; }
.meta { color: #666; font-size: 13px; margin-bottom: 20px; }
table { border-collapse: collapse; margin-bottom: 8px; }
th, td { border: 1px solid #ccc; padding: 6px 10px; font-size: 13px;
         vertical-align: middle; }
th { background: #f4f4f4; text-align: left; }
table.summary td.num { font-family: monospace; text-align: right; }
table.summary tr.agg { background: #fffbe6; font-weight: bold; }
table.samples img { display: block; image-rendering: pixelated; height: 48px; }
table.samples td.text { font-family: monospace; font-size: 13px;
                        white-space: pre; }
table.samples td.text span.bad { color: #d00; font-weight: bold; }
table.samples td.text span.good { color: #0a0; font-weight: bold; }
"""


def _png_b64(img_np):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _img_tag(img_np, alt=''):
    return (f'<img src="data:image/png;base64,{_png_b64(img_np)}" '
            f'alt="{html.escape(alt)}"/>')


def _diff_recon_html(recon, gt, masked=''):
    """Color recon chars relative to gt + the input mask view:
      - mismatch (any position) -> red (`bad`)
      - match AND position was masked (masked[i] == '_') -> green (`good`)
      - match AND not masked -> plain
    `masked` is the masked-input render; '_' marks masked positions."""
    out = []
    for i, ch in enumerate(recon):
        gt_ch = gt[i] if i < len(gt) else ''
        was_masked = i < len(masked) and masked[i] == '_'
        esc = html.escape(ch)
        if ch != gt_ch:
            out.append(f'<span class="bad">{esc}</span>')
        elif was_masked:
            out.append(f'<span class="good">{esc}</span>')
        else:
            out.append(esc)
    return ''.join(out)


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--output', type=str, default=None,
        help='HTML output path. Default: <pretrained_dir>/phase1_report.html')
    parser.add_argument(
        '--datasets', type=str, default=None,
        help='Comma-separated dataset basenames to include '
             '(e.g., "CUTE80,IIIT5k"). Default: all in cfg.')
    return parser.parse_args()


def _close_lmdb_envs(dataloader):
    if dataloader is None:
        return
    ds = getattr(dataloader, 'dataset', None)
    sets = getattr(ds, 'lmdb_sets', None) if ds is not None else None
    if not sets:
        return
    for v in sets.values():
        env = v.get('env') if isinstance(v, dict) else None
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main():
    flags = parse_args()
    assert flags.config is not None, 'Please specify -c <config_path>.'
    cfg = Config(flags.config)
    cfg.merge_dict(flags.opt)

    output_dir = (cfg.cfg['Global'].get('output_dir') or '').rstrip('/')
    cfg.cfg['Global']['output_dir'] = output_dir
    if cfg.cfg['Global'].get('pretrained_model') is None:
        cfg.cfg['Global']['pretrained_model'] = output_dir + '/best.pth'
    cfg.cfg['Global']['use_amp'] = False

    eval_ds = cfg.cfg['Eval']['dataset']
    all_dirs = [os.path.expanduser(p) for p in eval_ds.get(
        'data_dir_list', [eval_ds.get('data_dir', '')])]
    if flags.datasets:
        wanted = set(flags.datasets.split(','))
        all_dirs = [p for p in all_dirs
                    if p.rstrip('/').split('/')[-1] in wanted]
    assert all_dirs, 'No eval datasets resolved.'

    trainer = Trainer(cfg, mode='eval')

    per_dataset = []
    for ds_path in all_dirs:
        ds_name = ds_path.rstrip('/').split('/')[-1]
        # lmdb-py refuses to open the same path twice in one process. The
        # initial Trainer() built a dataloader covering all 6 dirs, and each
        # iter also leaves its single-dir dl behind — close their envs before
        # rebuilding so the next open() doesn't collide.
        _close_lmdb_envs(trainer.valid_dataloader)
        eval_ds['data_dir_list'] = [ds_path]
        trainer.valid_dataloader = build_dataloader(
            trainer.cfg, 'Eval', trainer.logger, task='rec')
        trainer.logger.info(f'[{ds_name}] eval ...')
        metric = trainer.eval()
        viz = metric.pop('_phase1_viz', [])
        metric.pop('_wandb_samples', None)
        per_dataset.append((ds_name, metric, viz))
        trainer.logger.info(
            f'[{ds_name}] linguistic_acc={metric.get("linguistic_acc", 0):.4f} '
            f'visual_mse={metric.get("visual_mse", 0):.4f} '
            f'#tok={int(metric.get("num_masked_tokens", 0) or 0)} '
            f'#pat={int(metric.get("num_masked_patches", 0) or 0)}')

    out_path = flags.output
    if out_path is None:
        pretrained_dir = os.path.dirname(cfg.cfg['Global']['pretrained_model'])
        out_path = os.path.join(pretrained_dir, 'phase1_report.html')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    _write_html(out_path, cfg.cfg['Global']['pretrained_model'], per_dataset)
    trainer.logger.info(f'Wrote phase 1 report -> {out_path}')


def _write_html(path, ckpt_path, per_dataset):
    # Weighted aggregates reconstructed from (ratio, count). The metric
    # already exposes num_masked_tokens / num_masked_patches, so we can
    # recover correct/SSE without re-running.
    tot_ling_correct = 0
    tot_ling_total = 0
    tot_vis_sse = 0.0
    tot_vis_count = 0
    for _, m, _ in per_dataset:
        n_tok = int(m.get('num_masked_tokens', 0) or 0)
        n_pat = int(m.get('num_masked_patches', 0) or 0)
        tot_ling_correct += int(round(m.get('linguistic_acc', 0.0) * n_tok))
        tot_ling_total += n_tok
        tot_vis_sse += m.get('visual_mse', 0.0) * n_pat
        tot_vis_count += n_pat
    eps = 1e-8
    agg_ling = tot_ling_correct / (tot_ling_total + eps)
    agg_vis = tot_vis_sse / (tot_vis_count + eps)

    rows_summary = []
    for name, m, _ in per_dataset:
        rows_summary.append(
            f'<tr><td>{html.escape(name)}</td>'
            f'<td class="num">{m.get("linguistic_acc", 0.0):.4f}</td>'
            f'<td class="num">{m.get("visual_mse", 0.0):.4f}</td>'
            f'<td class="num">{int(m.get("num_masked_tokens", 0) or 0)}</td>'
            f'<td class="num">{int(m.get("num_masked_patches", 0) or 0)}</td>'
            f'<td class="num">{m.get("fps", 0.0):.1f}</td>'
            f'</tr>')
    rows_summary.append(
        f'<tr class="agg"><td>Weighted overall</td>'
        f'<td class="num">{agg_ling:.4f}</td>'
        f'<td class="num">{agg_vis:.4f}</td>'
        f'<td class="num">{tot_ling_total}</td>'
        f'<td class="num">{tot_vis_count}</td>'
        f'<td class="num">-</td>'
        f'</tr>')

    sample_sections = []
    for name, _, viz in per_dataset:
        if not viz:
            sample_sections.append(
                f'<h3>{html.escape(name)}</h3>'
                f'<p><em>no samples</em></p>')
            continue
        rows = []
        for s in viz:
            rows.append(
                '<tr>'
                f'<td>{_img_tag(s["image_orig"])}</td>'
                f'<td>{_img_tag(s["image_masked"])}</td>'
                f'<td>{_img_tag(s["image_recon"])}</td>'
                f'<td class="text">{html.escape(s.get("gt_text", ""))}</td>'
                f'<td class="text">{html.escape(s.get("masked_text", ""))}</td>'
                f'<td class="text">{_diff_recon_html(s.get("recon_text", ""), s.get("gt_text", ""), s.get("masked_text", ""))}</td>'
                '</tr>')
        sample_sections.append(
            f'<h3>{html.escape(name)}</h3>'
            '<table class="samples">'
            '<tr><th>orig</th><th>masked</th><th>reconstructed</th>'
            '<th>gt_text</th><th>masked_text</th><th>recon_text</th></tr>'
            f'{"".join(rows)}</table>')

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>VL-Reader Phase 1 Report</title>
<style>{HTML_CSS}</style>
</head><body>
<h1>VL-Reader Phase 1 Reconstruction Report</h1>
<div class="meta">
  Checkpoint: <code>{html.escape(ckpt_path)}</code><br>
  Generated: {now}
</div>

<h2>Summary</h2>
<table class="summary">
<tr><th>dataset</th><th>linguistic_acc</th><th>visual_mse</th>
    <th>#masked tokens</th><th>#masked patches</th><th>fps</th></tr>
{"".join(rows_summary)}
</table>

<h2>Reconstruction samples</h2>
{"".join(sample_sections)}
</body></html>
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)


if __name__ == '__main__':
    main()
