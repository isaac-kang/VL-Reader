"""Masked Visual-Linguistic Decoder (MVLD) for VL-Reader.

The decoder consumes the encoder dict ``{feature, mask, target_patches}``
and the data tuple ``(labels, lengths)``. It runs ``dec_depth`` MVLD layers
that perform a tightly coupled vision/language exchange:

    H_v = MHA(F_v, F_v, F_v)              # visual self-attention
    H_q = MHA(F_q, F_l, F_l, m_ql)        # query-text cross-attention
    F_v = MHA(H_v, H_q, H_q)              # vision-query cross-attention
    F_q = MHA(H_q, H_v, H_v)              # query-vision cross-attention

In phase ``pretrain`` (MVLR) random visual patches are masked by the
encoder and random text tokens are masked here; both are reconstructed at
every decoder layer (MSE for pixels, CE for characters). In phase
``finetune`` no positions are masked, the visual head is disabled and
PARSeq-style permutation language modeling supervises only the linguistic
output. Inference follows PARSeq: AR decoding plus optional cloze-mask
refinement.
"""

import math
from itertools import permutations
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class MVLDLayer(nn.Module):
    """A single MVLD layer with the four attentions described in the paper."""

    def __init__(self, dim, nhead, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # Pre-LayerNorm before each attention input
        self.norm_v_self = nn.LayerNorm(dim)
        self.norm_q_text_q = nn.LayerNorm(dim)
        self.norm_q_text_kv = nn.LayerNorm(dim)
        self.norm_v_query_q = nn.LayerNorm(dim)
        self.norm_v_query_kv = nn.LayerNorm(dim)
        self.norm_q_vision_q = nn.LayerNorm(dim)
        self.norm_q_vision_kv = nn.LayerNorm(dim)

        self.v_self_attn = nn.MultiheadAttention(dim,
                                                 nhead,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.q_text_attn = nn.MultiheadAttention(dim,
                                                 nhead,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.v_query_attn = nn.MultiheadAttention(dim,
                                                  nhead,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.q_vision_attn = nn.MultiheadAttention(dim,
                                                   nhead,
                                                   dropout=dropout,
                                                   batch_first=True)

        # Shared FFN: same weights applied to both visual and query streams.
        # Forces a unified non-linear transformation across modalities, which
        # supports VL-Reader's cross-modal alignment objective. LayerNorms
        # remain stream-specific so each modality is normalized to its own
        # distribution before entering the shared FFN.
        hidden = int(dim * mlp_ratio)
        self.norm_v_ffn = nn.LayerNorm(dim)
        self.norm_q_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim),
                                 nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, fv, fq, fl, m_ql=None, m_ql_kpm=None, update_v=True):
        # Visual self-attention
        v_norm = self.norm_v_self(fv)
        v_attn, _ = self.v_self_attn(v_norm, v_norm, v_norm, need_weights=False)
        h_v = fv + self.drop(v_attn)

        # Query <- Text cross-attention (uses permuted/masked attention mask).
        # Cast key_padding_mask to the same dtype as attn_mask to avoid the
        # PyTorch warning about mismatched mask types.
        if m_ql_kpm is not None and m_ql is not None:
            m_ql_kpm = torch.zeros_like(m_ql_kpm,
                                        dtype=m_ql.dtype).masked_fill(
                                            m_ql_kpm, float('-inf'))
        q_norm = self.norm_q_text_q(fq)
        l_norm = self.norm_q_text_kv(fl)
        q_attn, _ = self.q_text_attn(q_norm,
                                     l_norm,
                                     l_norm,
                                     attn_mask=m_ql,
                                     key_padding_mask=m_ql_kpm,
                                     need_weights=False)
        h_q = fq + self.drop(q_attn)

        # Query <- Vision cross-attention
        q_v_norm = self.norm_q_vision_q(h_q)
        v_kv = self.norm_q_vision_kv(h_v)
        q_v_attn2, _ = self.q_vision_attn(q_v_norm,
                                          v_kv,
                                          v_kv,
                                          need_weights=False)
        fq_new = h_q + self.drop(q_v_attn2)
        fq_new = fq_new + self.drop(self.ffn(self.norm_q_ffn(fq_new)))

        if update_v:
            # Visual <- Query cross-attention
            v_q_norm = self.norm_v_query_q(h_v)
            q_v_kv = self.norm_v_query_kv(h_q)
            v_q_attn, _ = self.v_query_attn(v_q_norm,
                                            q_v_kv,
                                            q_v_kv,
                                            need_weights=False)
            fv_new = h_v + self.drop(v_q_attn)
            fv_new = fv_new + self.drop(self.ffn(self.norm_v_ffn(fv_new)))
        else:
            fv_new = h_v
        return fv_new, fq_new


class MVLDecoder(nn.Module):

    def __init__(self,
                 in_channels=768,
                 out_channels=37,
                 max_label_length=25,
                 embed_dim=768,
                 dec_num_heads=12,
                 dec_mlp_ratio=4,
                 dec_depth=4,
                 patch_size=(4, 8),
                 in_image_channels=3,
                 perm_num=6,
                 perm_forward=True,
                 perm_mirrored=True,
                 text_mask_ratio=0.2,
                 lambda_v=1.0,
                 lambda_l=1.0,
                 phase='pretrain',
                 decode_ar=True,
                 refine_iters=1,
                 dropout=0.1,
                 **kwargs: Any) -> None:
        super().__init__()
        assert phase in ('pretrain', 'finetune')
        self.phase = phase
        self.pad_id = out_channels - 1
        self.eos_id = 0
        self.bos_id = out_channels - 2
        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters
        self.text_mask_ratio = text_mask_ratio
        self.lambda_v = lambda_v
        self.lambda_l = lambda_l
        self.in_image_channels = in_image_channels
        self.patch_size = list(patch_size)
        self.patch_dim = self.patch_size[0] * self.patch_size[
            1] * in_image_channels

        self.layers = nn.ModuleList([
            MVLDLayer(embed_dim,
                      dec_num_heads,
                      mlp_ratio=dec_mlp_ratio,
                      dropout=dropout) for _ in range(dec_depth)
        ])

        # Heads (applied after each layer per Eq. 7-8)
        self.head_v_norm = nn.LayerNorm(embed_dim)
        self.head_v = nn.Linear(embed_dim, self.patch_dim)
        self.head_l_norm = nn.LayerNorm(embed_dim)
        # Don't predict <bos> nor <pad>
        self.head_l = nn.Linear(embed_dim, out_channels - 2)

        # Text embedding shared with PARSeq's convention.
        self.text_embed = TokenEmbedding(out_channels, embed_dim)

        # Learnable [MASK_l] token used to replace masked input characters.
        self.mask_l_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position encodings for text input and query tokens (length L+1).
        self.pos_text = nn.Parameter(
            torch.zeros(1, max_label_length + 1, embed_dim))
        self.pos_queries = nn.Parameter(
            torch.zeros(1, max_label_length + 1, embed_dim))

        self.dropout = nn.Dropout(p=dropout)

        # Permutation parameters (only used in fine-tune phase)
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        nn.init.trunc_normal_(self.pos_text, std=0.02)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        nn.init.trunc_normal_(self.mask_l_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'text_embed.embedding.weight', 'pos_text', 'pos_queries',
            'mask_l_token'
        }

    # -- helpers ----------------------------------------------------------------

    def _embed_text(self, tokens, replace_mask=None):
        """Embed text tokens; optionally replace selected positions with [MASK_l]."""
        L = tokens.shape[1]
        emb = self.text_embed(tokens)
        if replace_mask is not None:
            emb = torch.where(replace_mask.unsqueeze(-1), self.mask_l_token,
                              emb)
        emb = emb + self.pos_text[:, :L]
        return self.dropout(emb)

    def _query_tokens(self, bs, num_steps, device):
        return self.dropout(self.pos_queries[:, :num_steps].expand(bs, -1, -1))

    def _build_layers(self,
                      fv_in,
                      fq_in,
                      fl,
                      m_ql=None,
                      m_ql_kpm=None,
                      collect_visual=True):
        fv = fv_in
        fq = fq_in
        v_logits, l_logits = [], []
        last_idx = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            # The visual stream at the final layer is only useful when we
            # collect a visual reconstruction from it; otherwise skip it
            # (PARSeq-style "don't update content at last layer").
            update_v = collect_visual or i != last_idx
            fv, fq = layer(fv,
                           fq,
                           fl,
                           m_ql=m_ql,
                           m_ql_kpm=m_ql_kpm,
                           update_v=update_v)
            if collect_visual:
                v_logits.append(self.head_v(self.head_v_norm(fv)))
            l_logits.append(self.head_l(self.head_l_norm(fq)))
        return fv, fq, v_logits, l_logits

    # -- training: phase 1 (pretrain MVLR) -------------------------------------

    def _phase1_step(self, encoder_out, data):
        feature = encoder_out['feature']  # (B, N, C) — visual w/ mask tokens
        v_mask = encoder_out['mask']  # (B, N) — True at masked patches
        target_patches = encoder_out['target_patches']  # (B, N, patch_dim)

        # Phase 1 = PARSeq-style causal AR with selective loss:
        #   - text input fl has L+1 tokens [BOS, c_1..c_L]; some replaced by
        #     [MASK_l]
        #   - causal forward attention mask (q_k attends to keys 0..k)
        #   - kpm blocks attention to [MASK_l] positions and PADs (prevents
        #     leakage of the mask-token embedding into other queries)
        #   - supervision (selective CE) only at output positions k where
        #     INPUT position k was masked, target = tgt_out[k] = c_{k+1}.
        #     Following Fig. 4 of the paper: when c_p is masked at input
        #     col p, q_p (which targets c_{p+1}) is the query whose context
        #     is degraded — the loss trains the model to predict the next
        #     character despite that hole.
        tgt = data[0].long()  # (B, L+2) = [BOS, c1..cL, EOS, PAD..]
        bs = tgt.shape[0]
        device = tgt.device
        tgt_in = tgt[:, :-1]  # (B, L+1)
        tgt_out = tgt[:, 1:]  # (B, L+1)

        maskable = (tgt_in != self.bos_id) & (tgt_in != self.eos_id) & (
            tgt_in != self.pad_id)
        rand = torch.rand_like(maskable, dtype=torch.float32)
        text_mask = (rand < self.text_mask_ratio) & maskable  # (B, L+1)

        fl = self._embed_text(tgt_in, replace_mask=text_mask)

        L_q = tgt_in.shape[1]
        fq_in = self._query_tokens(bs, L_q, device)

        m_ql = self._forward_attn_mask(L_q, device)  # causal forward
        pad_mask = (tgt_in == self.pad_id)
        m_ql_kpm = text_mask | pad_mask

        fv, fq, v_logits, l_logits = self._build_layers(feature,
                                                        fq_in,
                                                        fl,
                                                        m_ql=m_ql,
                                                        m_ql_kpm=m_ql_kpm,
                                                        collect_visual=True)

        # Visual reconstruction loss: averaged across layers and masked patches.
        v_loss = fq.new_zeros(())
        if v_mask is not None and v_mask.any():
            for v_pred in v_logits:
                diff = (v_pred - target_patches).pow(2).mean(dim=-1)
                v_loss = v_loss + (diff * v_mask.float()).sum() / v_mask.sum()
            v_loss = v_loss / max(1, len(v_logits))

        # Linguistic reconstruction loss: averaged across layers, selective CE
        # at output positions where the corresponding INPUT position was masked.
        l_loss = fq.new_zeros(())
        ml = text_mask  # (B, L+1) — input-position mask, used as-is
        n_masked = ml.sum().item()
        if n_masked > 0:
            n_classes = self.head_l.out_features
            tgt_flat = tgt_out.reshape(-1)
            ml_flat = ml.reshape(-1)
            for q_pred in l_logits:
                logits_flat = q_pred.reshape(-1, n_classes)
                logits_sel = logits_flat[ml_flat]
                tgt_sel = tgt_flat[ml_flat]
                l_loss = l_loss + F.cross_entropy(
                    logits_sel, tgt_sel, ignore_index=self.pad_id)
            l_loss = l_loss / max(1, len(l_logits))

        loss = self.lambda_v * v_loss + self.lambda_l * l_loss
        return [loss, l_logits[-1]]

    # -- eval: phase 1 (MVLR reconstruction metrics + viz dump) ----------------

    def _phase1_eval(self, encoder_out, data):
        """Phase 1 eval: same forward as training, return predictions+masks dict
        (no AR generation). Used by VLReaderPhase1Metric and the visualizer.
        """
        feature = encoder_out['feature']
        v_mask = encoder_out['mask']
        target_patches = encoder_out['target_patches']

        if data is None:
            # No labels: visual recon only.
            fq_in = self._query_tokens(feature.shape[0],
                                       self.max_label_length + 1,
                                       feature.device)
            fl = self.dropout(self.pos_text.expand(feature.shape[0], -1, -1))
            _, _, v_logits, _ = self._build_layers(feature,
                                                   fq_in,
                                                   fl,
                                                   collect_visual=True)
            return {
                'visual_pred': v_logits[-1],
                'visual_mask': v_mask,
                'target_patches': target_patches,
                'linguistic_pred': None,
                'text_mask': None,
                'target_text': None,
                'tgt_in': None,
            }

        tgt = data[0].long()
        bs = tgt.shape[0]
        device = tgt.device
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        L_q = tgt_in.shape[1]

        maskable = (tgt_in != self.bos_id) & (tgt_in != self.eos_id) & (
            tgt_in != self.pad_id)
        rand = torch.rand_like(maskable, dtype=torch.float32)
        text_mask = (rand < self.text_mask_ratio) & maskable

        fl = self._embed_text(tgt_in, replace_mask=text_mask)
        fq_in = self._query_tokens(bs, L_q, device)

        m_ql = self._forward_attn_mask(L_q, device)
        pad_mask = (tgt_in == self.pad_id)
        m_ql_kpm = text_mask | pad_mask

        _, _, v_logits, l_logits = self._build_layers(feature,
                                                      fq_in,
                                                      fl,
                                                      m_ql=m_ql,
                                                      m_ql_kpm=m_ql_kpm,
                                                      collect_visual=True)
        return {
            'visual_pred': v_logits[-1],         # (B, N, patch_dim)
            'visual_mask': v_mask,                # (B, N) bool
            'target_patches': target_patches,    # (B, N, patch_dim)
            'target_mean': encoder_out.get('target_mean'),
            'target_var': encoder_out.get('target_var'),
            'linguistic_pred': l_logits[-1],     # (B, L+1, n_classes)
            'text_mask': text_mask,               # (B, L+1) bool
            'target_text': tgt_out,               # (B, L+1) long
            'tgt_in': tgt_in,                     # (B, L+1) long
        }

    # -- training: phase 2 (fine-tune with PLM) --------------------------------

    def _phase2_step(self, encoder_out, data):
        feature = encoder_out['feature']
        tgt = data[0].long()
        bs = tgt.shape[0]
        device = tgt.device
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        L_q = tgt_in.shape[1]

        fl = self._embed_text(tgt_in, replace_mask=None)
        fq_in = self._query_tokens(bs, L_q, device)

        # Permutations (PARSeq style) over the maximum content length in batch.
        tgt_perms = self._gen_tgt_perms(tgt, device)

        # Padding mask for keys: ignore PAD positions in text.
        kpm = (tgt_in == self.pad_id)

        loss = fq_in.new_zeros(())
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        first_logits = None
        for i, perm in enumerate(tgt_perms):
            query_mask = self._generate_attn_mask(perm, device)
            _, _, _, l_logits = self._build_layers(feature,
                                                   fq_in,
                                                   fl,
                                                   m_ql=query_mask,
                                                   m_ql_kpm=kpm,
                                                   collect_visual=False)
            logits = l_logits[-1]
            if first_logits is None:
                first_logits = logits
            loss = loss + n * F.cross_entropy(
                logits.flatten(end_dim=1),
                tgt_out.flatten(),
                ignore_index=self.pad_id)
            loss_numel += n
            if i == 1:
                # After forward+reverse perms, drop EOS supervision for further perms.
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id,
                                      tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss = loss / max(1, loss_numel)
        return [loss, first_logits]

    # -- inference --------------------------------------------------------------

    def _decode_step(self,
                     fv,
                     tgt_in,
                     num_steps,
                     query_mask=None,
                     kpm=None):
        """Run all MVLD layers and return last-layer linguistic logits."""
        bs = fv.shape[0]
        device = fv.device
        fl = self._embed_text(tgt_in, replace_mask=None)
        fq_in = self._query_tokens(bs, num_steps, device)
        _, fq, _, _ = self._build_layers(fv,
                                         fq_in,
                                         fl,
                                         m_ql=query_mask,
                                         m_ql_kpm=kpm,
                                         collect_visual=False)
        return self.head_l(self.head_l_norm(fq))

    def _forward_test(self, encoder_out, max_length=None):
        feature = encoder_out['feature']
        bs = feature.shape[0]
        device = feature.device
        max_length = (self.max_label_length if max_length is None else min(
            max_length, self.max_label_length))
        num_steps = max_length + 1  # include EOS slot

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps),
                                self.pad_id,
                                dtype=torch.long,
                                device=device)
            tgt_in[:, 0] = self.bos_id

            # Pre-build the canonical forward attention mask for the full length.
            full_mask = self._forward_attn_mask(num_steps, device)
            logits_all = []
            for i in range(num_steps):
                j = i + 1
                # Build the mask sliced to current length
                step_mask = full_mask[:j, :j]
                # Run the decoder using only the j-prefix of tgt_in.
                step_logits = self._decode_step(feature,
                                                tgt_in[:, :j],
                                                num_steps=j,
                                                query_mask=step_mask)
                # The new prediction is the last output position
                p_i = step_logits[:, i:j, :]
                logits_all.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze(1).argmax(-1)
                    if (tgt_in == self.eos_id).any(dim=-1).all():
                        # Pad remaining and break early
                        break

            logits = torch.cat(logits_all, dim=1)
            # Pad the logits to num_steps if we exited early.
            if logits.shape[1] < num_steps:
                pad_n = num_steps - logits.shape[1]
                pad_logits = logits.new_full(
                    (bs, pad_n, logits.shape[-1]), 0.0)
                # Predict PAD = eos_id index argmax doesn't matter; keep zeros.
                logits = torch.cat([logits, pad_logits], dim=1)
        else:
            # Single-shot prediction with all queries; only BOS as context.
            tgt_in = torch.full((bs, num_steps),
                                self.pad_id,
                                dtype=torch.long,
                                device=device)
            tgt_in[:, 0] = self.bos_id
            cloze = self._cloze_attn_mask(num_steps, device)
            logits = self._decode_step(feature,
                                       tgt_in,
                                       num_steps=num_steps,
                                       query_mask=cloze)

        # Refinement using cloze mask over the predicted sequence.
        if self.refine_iters:
            cloze = self._cloze_attn_mask(num_steps, device)
            bos = torch.full((bs, 1),
                             self.bos_id,
                             dtype=torch.long,
                             device=device)
            for _ in range(self.refine_iters):
                pred = logits.argmax(-1)
                tgt_in = torch.cat([bos, pred[:, :-1]], dim=1)
                # Block attention to positions beyond the first EOS in the
                # predicted sequence.
                kpm = (tgt_in == self.eos_id).int().cumsum(-1) > 0
                logits = self._decode_step(feature,
                                           tgt_in,
                                           num_steps=num_steps,
                                           query_mask=cloze,
                                           kpm=kpm)

        return F.softmax(logits, -1)

    # -- attention masks --------------------------------------------------------

    def _forward_attn_mask(self, sz, device):
        """Standard causal mask of shape (sz, sz) — query i sees keys 0..i.

        Used when no permutation language model is active. Returns an
        additive mask compatible with ``nn.MultiheadAttention``.
        """
        mask = torch.zeros(sz, sz, device=device)
        mask = mask.masked_fill(
            torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), 1),
            float('-inf'))
        return mask

    def _cloze_attn_mask(self, sz, device):
        """Cloze mask: query i sees all keys except key i (no self-leak)."""
        mask = torch.zeros(sz, sz, device=device)
        mask[torch.eye(sz, dtype=torch.bool, device=device)] = float('-inf')
        return mask

    def _gen_tgt_perms(self, tgt, device):
        """PARSeq-style permutation generation."""
        max_num_chars = tgt.shape[1] - 2
        if max_num_chars == 1:
            return torch.arange(3, device=device).unsqueeze(0)
        perms = ([torch.arange(max_num_chars, device=device)]
                 if self.perm_forward else [])
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(
                permutations(range(max_num_chars), max_num_chars)),
                                        device=device)[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                idx = self.rng.choice(len(perm_pool),
                                      size=num_gen_perms - len(perms),
                                      replace=False)
                perms = torch.cat([perms, perm_pool[idx]])
        else:
            perms.extend([
                torch.randperm(max_num_chars, device=device)
                for _ in range(num_gen_perms - len(perms))
            ])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            comp = perms.flip(-1)
            perms = torch.stack([perms, comp
                                 ]).transpose(0, 1).reshape(-1, max_num_chars)
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1,
                                                            device=device)
        return perms

    def _generate_attn_mask(self, perm, device):
        """Generate the (L_q, L_q) query-text attention mask for one permutation.

        ``perm`` is a permutation over [0, L+1] inclusive of BOS and EOS
        positions. The returned mask is additive: 0 where attention is
        allowed, -inf where it's blocked. Self attention is blocked too.
        """
        sz = perm.shape[0]
        mask = torch.zeros(sz, sz, device=device)
        for i in range(sz):
            qi = perm[i].item()
            masked_keys = perm[i + 1:]
            mask[qi, masked_keys] = float('-inf')
        # Block self-attention (no self-leak)
        mask[torch.eye(sz, dtype=torch.bool,
                       device=device)] = float('-inf')
        # Strip BOS row and EOS column to get (L+1, L+1)
        return mask[1:, :-1]

    # -- entrypoint -------------------------------------------------------------

    def forward(self, x, data=None):
        # ``x`` may either be the encoder dict or a bare tensor (legacy).
        if isinstance(x, dict):
            encoder_out = x
        else:
            encoder_out = {
                'feature': x,
                'mask': None,
                'target_patches': None,
            }

        if self.training:
            if self.phase == 'pretrain':
                return self._phase1_step(encoder_out, data)
            return self._phase2_step(encoder_out, data)
        # Eval mode
        if self.phase == 'pretrain':
            return self._phase1_eval(encoder_out, data)
        return self._forward_test(encoder_out)
