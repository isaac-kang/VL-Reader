"""VL-Reader visual encoder.

ViT encoder with MAE-style random patch masking. During phase 1 (MVLR
pre-training), a fraction ``mask_ratio`` of image patches are dropped before
being processed by the ViT blocks; their positions are filled with a
learnable visual mask token after encoding so that the masked visual
linguistic decoder receives the full spatial sequence. During phase 2 the
mask ratio is set to 0 and the encoder degrades to a vanilla ViT.

The encoder also returns the per-patch reconstruction targets and the
per-patch boolean mask so that the loss can supervise visual reconstruction
on the masked positions only.
"""

import numpy as np
import torch
from torch import nn
from torch.nn.init import ones_, trunc_normal_, zeros_

from openrec.modeling.common import Block, PatchEmbed


class ViTVLReader(nn.Module):

    def __init__(
        self,
        img_size=[32, 128],
        patch_size=[4, 8],
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        mask_ratio=0.75,
        norm_pix_loss=True,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_channels = embed_dim
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_dim = patch_size[0] * patch_size[1] * in_channels

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels,
                                      embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # Shared learnable visual mask token [MASK_v]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(
            torch.zeros([1, num_patches, embed_dim], dtype=torch.float32))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=act_layer,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, mean=0, std=0.02)
        trunc_normal_(self.mask_token, mean=0, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'mask_token'}

    def patchify(self, imgs):
        """Convert (B, C, H, W) image batch into per-patch ground-truth values.

        Returns a tensor of shape (B, num_patches, patch_h * patch_w * C).
        """
        ph, pw = self.patch_size
        b, c, h, w = imgs.shape
        assert h % ph == 0 and w % pw == 0
        nh, nw = h // ph, w // pw
        x = imgs.reshape(b, c, nh, ph, nw, pw)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(b, nh * nw, ph * pw * c)
        return x

    def random_masking(self, x, mask_ratio):
        """MAE-style random masking.

        ``x`` is the full patch-embedded sequence (B, N, C). Returns the
        kept (visible) tokens, a boolean mask (B, N) where True marks
        masked positions, and the index used to undo the random shuffle.
        """
        b, n, c = x.shape
        len_keep = max(1, int(n * (1 - mask_ratio)))

        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x,
                                 dim=1,
                                 index=ids_keep.unsqueeze(-1).expand(
                                     -1, -1, c))

        mask = torch.ones(b, n, device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = False
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_visible, mask, ids_restore

    def forward(self, imgs):
        # Raw per-patch pixels (un-normalized) — kept around for visualization.
        target_patches_raw = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target_patches_raw.mean(dim=-1, keepdim=True)
            var = target_patches_raw.var(dim=-1, keepdim=True)
            target_patches = (target_patches_raw - mean) / (var + 1.0e-6)**0.5
        else:
            mean = target_patches_raw.new_zeros(
                target_patches_raw.shape[:-1] + (1, ))
            var = target_patches_raw.new_ones(
                target_patches_raw.shape[:-1] + (1, ))
            target_patches = target_patches_raw

        x = self.patch_embed(imgs)
        x = x + self.pos_embed

        # Apply masking whenever mask_ratio>0 (training AND eval). Phase 2 sets
        # mask_ratio=0 so inference there bypasses this. For phase 1 eval we
        # want masking applied so the reconstruction objective can be measured;
        # determinism is the trainer's job (it seeds before each eval pass).
        if self.mask_ratio > 0:
            x_visible, mask, ids_restore = self.random_masking(
                x, self.mask_ratio)
            x_visible = self.pos_drop(x_visible)
            for blk in self.blocks:
                x_visible = blk(x_visible)
            x_visible = self.norm(x_visible)

            b, n_full, c = x.shape
            n_keep = x_visible.shape[1]
            mask_tokens = self.mask_token.expand(b, n_full - n_keep, c)
            x_full = torch.cat([x_visible, mask_tokens], dim=1)
            x_full = torch.gather(x_full,
                                  dim=1,
                                  index=ids_restore.unsqueeze(-1).expand(
                                      -1, -1, c))
            # Visible positions already carry pos_embed (added before masking
            # and processed by ViT). Add pos_embed only at masked positions so
            # the [MASK_v] token gains spatial information.
            x_full = x_full + self.pos_embed * mask.float().unsqueeze(-1)
        else:
            x = self.pos_drop(x)
            for blk in self.blocks:
                x = blk(x)
            x_full = self.norm(x)
            mask = torch.zeros(x_full.shape[0],
                               x_full.shape[1],
                               device=x_full.device,
                               dtype=torch.bool)

        return {
            'feature': x_full,
            'mask': mask,
            'target_patches': target_patches,
            'target_patches_raw': target_patches_raw,
            'target_mean': mean,
            'target_var': var,
        }
