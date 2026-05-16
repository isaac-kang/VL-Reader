"""Metric for VL-Reader phase 1.

Tracks two reconstruction metrics that match the MVLR training objective:

- ``linguistic_acc`` (primary): top-1 character accuracy on masked text
  positions. This is what the model is actually trained for, unlike full
  AR generation which phase 1 never sees during training.
- ``visual_mse``: mean squared error between predicted and ground-truth
  pixels at the masked image patches (computed in the same normalized
  space as the loss, so comparable across runs).

The metric is fed the dict that VLReaderPhase1PostProcess passes through.
"""

import torch


class VLReaderPhase1Metric:

    def __init__(self,
                 main_indicator='linguistic_acc',
                 ignore_index=None,
                 **kwargs):
        self.main_indicator = main_indicator
        self.ignore_index = ignore_index
        self.eps = 1e-8
        self.reset()

    def __call__(self, post_result, batch=None, training=False, *args,
                 **kwargs):
        # ``post_result`` is the pass-through dict from
        # VLReaderPhase1PostProcess. We accumulate sums and counts here so
        # ``get_metric`` can return averages over the whole eval set.
        if not isinstance(post_result, dict):
            return self._snapshot()

        # Linguistic: top-1 acc on masked tokens
        ling = post_result.get('linguistic_pred')
        text_mask = post_result.get('text_mask')
        target = post_result.get('target_text')
        if ling is not None and text_mask is not None and target is not None:
            with torch.no_grad():
                pred_ids = ling.argmax(-1)
                correct = (pred_ids == target) & text_mask
                self.ling_correct += int(correct.sum().item())
                self.ling_total += int(text_mask.sum().item())

        # Visual: MSE on masked patches (normalized space)
        vp = post_result.get('visual_pred')
        vmask = post_result.get('visual_mask')
        tgt_p = post_result.get('target_patches')
        if vp is not None and vmask is not None and tgt_p is not None and vmask.any():
            with torch.no_grad():
                diff = (vp - tgt_p).pow(2).mean(dim=-1)  # (B, N)
                self.vis_sse += float((diff * vmask.float()).sum().item())
                self.vis_count += int(vmask.sum().item())

        return self._snapshot()

    def _snapshot(self):
        return {
            'linguistic_acc': self.ling_correct /
            (self.ling_total + self.eps),
            'visual_mse': self.vis_sse / (self.vis_count + self.eps),
        }

    def get_metric(self, training=False):
        snap = self._snapshot()
        out = {
            'linguistic_acc': snap['linguistic_acc'],
            'visual_mse': snap['visual_mse'],
            'num_masked_tokens': self.ling_total,
            'num_masked_patches': self.vis_count,
        }
        self.reset()
        return out

    def reset(self):
        self.ling_correct = 0
        self.ling_total = 0
        self.vis_sse = 0.0
        self.vis_count = 0
