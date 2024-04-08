# code taken from https://github.com/yabufarha/ms-tcn
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltc.model.slot import SlotAttentionModule


class I3D(nn.Module):
    def __init__(self, model_cfg):
        super(I3D, self).__init__()

        self.conv1 = nn.Conv1d(2048, 512, 1)
        self.action_slot = SlotAttentionModule(model_cfg.NUM_CLASSES, 512)

    def forward(self, x, mask):
        x = F.relu(self.conv1(x)*mask)
        attn, x = self.action_slot.action_slot_forward(x, mask)
        x = x* masks[:, 0:1, :]
        return x