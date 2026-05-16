"""PostProcess for VL-Reader phase 1 (MVLR pre-training).

The decoder returns a dict of reconstruction predictions and masks instead
of character logits. This pass-through wrapper just decodes a few helper
strings (GT, masked, reconstructed text) for the visualizer and forwards
the rest of the dict to the metric and visualization hooks unchanged.
"""

from .ctc_postprocess import BaseRecLabelDecode


class VLReaderPhase1PostProcess(BaseRecLabelDecode):
    """Phase 1 post-process: decode text strings for the visualizer."""

    BOS = '<s>'
    EOS = '</s>'
    PAD = '<pad>'
    MASK_DISPLAY = '_'

    def __init__(self,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super().__init__(character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        # Match ARLabelEncode's order: [EOS, ...chars..., BOS, PAD]
        return [self.EOS] + dict_character + [self.BOS, self.PAD]

    def __call__(self, preds, batch=None, *args, **kwargs):
        # ``preds`` is a dict from MVLDecoder._phase1_eval. For loss/metric
        # consumers we pass it through; for the visualizer we add string
        # versions of GT / masked-input / reconstructed text on the side.
        if not isinstance(preds, dict):
            return preds

        # Pass-through preserves visual_pred / visual_mask / target_patches /
        # target_mean / target_var / linguistic_pred / text_mask / target_text
        # for the metric and the trainer's phase-1 visualizer.
        out = dict(preds)
        if preds.get('linguistic_pred') is not None and preds.get(
                'tgt_in') is not None:
            target_text = preds['target_text'].detach().cpu().tolist()
            text_mask_input = preds['text_mask'].detach().cpu().tolist()
            recon_ids = preds['linguistic_pred'].argmax(
                -1).detach().cpu().tolist()
            out['gt_text'] = self._ids_to_str(target_text)
            # masked_text shows the INPUT view: tgt_in (drop BOS via shift=True)
            # with `_` at masked input positions.
            out['masked_text'] = self._ids_to_str(
                preds['tgt_in'].detach().cpu().tolist(),
                mask=text_mask_input,
                shift=True,
            )
            # recon_text shows OUTPUT view: target_text with model's prediction
            # overlaid at the queries that were supervised. In phase 1 (B),
            # supervision is at output position k iff text_mask[k]=True (i.e.,
            # input position k was masked). The model's prediction at q_k
            # targets c_{k+1}. So overlay positions in recon_text are visually
            # offset by 1 from the `_` positions in masked_text — that's the
            # AR-shift artifact, not a bug.
            out['recon_text'] = self._ids_to_str(
                target_text,
                overlay=recon_ids,
                overlay_mask=text_mask_input)
        return out

    def _ids_to_str(self,
                    ids,
                    mask=None,
                    shift=False,
                    overlay=None,
                    overlay_mask=None):
        """Render token-id rows into strings.

        ``mask`` (B, L) bool: render MASK_DISPLAY at True positions.
        ``shift=True``: ids are tgt_in (BOS, c1..cL); drop BOS so output
        aligns with tgt_out strings.
        ``overlay`` (B, L) ids + ``overlay_mask`` (B, L) bool: where mask
        is True, replace the rendered char with the overlay's char (used to
        splice model predictions only at masked positions).
        """
        out = []
        for i, row in enumerate(ids):
            if shift:
                row = row[1:]
                row_mask = mask[i][1:] if mask is not None else None
                row_overlay = overlay[i][1:] if overlay is not None else None
                row_om = (overlay_mask[i][1:]
                          if overlay_mask is not None else None)
            else:
                row_mask = mask[i] if mask is not None else None
                row_overlay = overlay[i] if overlay is not None else None
                row_om = overlay_mask[i] if overlay_mask is not None else None
            chars = []
            for j, idx in enumerate(row):
                idx = int(idx)
                if idx >= len(self.character):
                    continue
                ch = self.character[idx]
                if ch == self.EOS:
                    break
                if ch in (self.BOS, self.PAD):
                    continue
                if row_om is not None and bool(row_om[j]):
                    o_idx = int(row_overlay[j])
                    if 0 <= o_idx < len(self.character):
                        ch = self.character[o_idx]
                        if ch in (self.EOS, self.BOS, self.PAD):
                            ch = '?'
                if row_mask is not None and bool(row_mask[j]):
                    chars.append(self.MASK_DISPLAY)
                else:
                    chars.append(ch)
            out.append(''.join(chars))
        return out
