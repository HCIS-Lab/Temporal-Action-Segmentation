# code taken from https://github.com/yabufarha/ms-tcn
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltc.model.slot import SlotAttentionModule


class MultiStageModel(nn.Module):
    def __init__(self, model_cfg):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(model_cfg.TCN.NUM_LAYERS,
                                       model_cfg.TCN.NUM_F_MAPS,
                                       model_cfg.INPUT_DIM,
                                       model_cfg.NUM_CLASSES,
                                       model_cfg.ACTION_SLOT,
                                       model_cfg.PROGRESS_SLOT)
        # self.stages = nn.ModuleList([
        #     copy.deepcopy(SingleStageModel(model_cfg.TCN.NUM_LAYERS,
        #                                    model_cfg.TCN.NUM_F_MAPS,
        #                                    model_cfg.NUM_CLASSES,
        #                                    model_cfg.NUM_CLASSES,
        #                                    model_cfg.TCN.SLOT)) for s in range(model_cfg.TCN.NUM_STAGES - 1)])
        self.stages = nn.ModuleList([
            SingleStageModel(model_cfg.TCN.NUM_LAYERS,
                                           model_cfg.TCN.NUM_F_MAPS,
                                           model_cfg.NUM_CLASSES,
                                           model_cfg.NUM_CLASSES,
                                           model_cfg.ACTION_SLOT,
                                           model_cfg.PROGRESS_SLOT) for s in range(model_cfg.TCN.NUM_STAGES - 1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        # action_attn_stack = None
        # if action_attn != None:
        #     action_attn_stack = action_attn.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            # if action_attn != None:
            #     action_attn_stack = torch.cat((action_attn_stack, action_attn.unsqueeze(0)), dim=0)
        # return outputs, action_attn_stack
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, num_action_slot=0, num_progress_slots=0):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.num_action_slot = num_action_slot
        self.num_progress_slots = num_progress_slots

        if num_action_slot > 0:
            self.num_action_slot = num_action_slot
            self.action_slot = SlotAttentionModule(num_action_slot, num_f_maps, num_classes)
        if num_progress_slots > 0:
            self.num_progress_slots = num_progress_slots
            self.progress_slot = SlotAttentionModule(num_progress_slots, num_f_maps, num_classes)

    def forward(self, x, masks):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, masks)
        if self.num_action_slot > 0:
            action_attn, out = self.action_slot.action_slot_forward(out, masks)
        if self.num_progress_slots > 0:
            progress_attn, progress_out = self.progress_slot.action_slot_forward(out, masks)
            out = ((self.conv_out(out) + progress_out)/2)
        if not self.num_action_slot > 0 and not self.num_progress_slots >0:
            out = self.conv_out(out) 
        out = out * masks[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x, masks):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        return (x + out) * masks


