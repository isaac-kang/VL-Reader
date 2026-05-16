"""Loss wrapper for VL-Reader.

The MVLR objective (visual MSE + linguistic CE) and the PARSeq-style PLM
cross-entropy are both computed inside ``MVLDecoder`` because they need
direct access to per-layer features and per-position masks. This module
unwraps the decoder return into the standard ``{'loss': ...}`` dict and
forwards optional detached component losses for logging.
"""

from torch import nn


class VLReaderLoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, predicts, batch):
        loss = predicts[0]
        out = {'loss': loss}
        if len(predicts) > 2 and isinstance(predicts[2], dict):
            for key, value in predicts[2].items():
                out[key] = value.detach()
        return out
