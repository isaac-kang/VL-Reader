"""Loss wrapper for VL-Reader.

The MVLR objective (visual MSE + linguistic CE) and the PARSeq-style PLM
cross-entropy are both computed inside ``MVLDecoder`` because they need
direct access to per-layer features and per-position masks. This module
just unwraps the ``[loss, logits]`` pair the decoder returns so that the
trainer sees the standard ``{'loss': ...}`` dict.
"""

from torch import nn


class VLReaderLoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, predicts, batch):
        loss, _ = predicts
        return {'loss': loss}
