from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from yacs.config import CfgNode
import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


class CEplusMSE(nn.Module):
    """
    Loss from MS-TCN paper. CrossEntropy + MSE
    https://arxiv.org/abs/1903.01945
    """
    def __init__(self, cfg: CfgNode):
        super(CEplusMSE, self).__init__()
        ignore_idx = cfg.MODEL.PAD_IGNORE_IDX
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        self.mse = nn.MSELoss(reduction='none')
        self.mse_fraction = cfg.MODEL.MSE_LOSS_FRACTION
        self.mse_clip_val = cfg.MODEL.MSE_LOSS_CLIP_VAL
        self.num_classes = cfg.MODEL.NUM_CLASSES

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        :param logits: [n_stages, batch_size, n_classes, seq_len]
        :param targets: [batch_size, seq_len]
        :return:
        """
        loss_dict = {"loss": 0.0, "loss_ce": 0.0, "loss_mse": 0.0}
        for p in logits:
            loss_dict['loss_ce'] += self.ce(rearrange(p, "b n_classes seq_len -> (b seq_len) n_classes"),
                                            rearrange(targets, "b seq_len -> (b seq_len)"))

            loss_dict['loss_mse'] += torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                                                     F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                                            min=0,
                                                            max=self.mse_clip_val))

        loss_dict['loss'] = loss_dict['loss_ce'] + self.mse_fraction * loss_dict['loss_mse']

        return loss_dict

class CEplusMSE_SlotBCE(nn.Module):
    """
    Loss from MS-TCN paper. CrossEntropy + MSE
    https://arxiv.org/abs/1903.01945
    """
    def __init__(self, cfg: CfgNode):
        super(CEplusMSE_SlotBCE, self).__init__()
        ignore_idx = cfg.MODEL.PAD_IGNORE_IDX
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        self.mse = nn.MSELoss(reduction='none')
        self.mask_ce = nn.CrossEntropyLoss()
        self.mse_fraction = cfg.MODEL.MSE_LOSS_FRACTION
        self.mask_ce_fraction = cfg.MODEL.MASK_LOSS_FRACTION
        self.mse_clip_val = cfg.MODEL.MSE_LOSS_CLIP_VAL
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.cfg = cfg
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        :param logits: [n_stages, batch_size, n_classes, seq_len]
        :param targets: [batch_size, seq_len]
        :return:
        """
        loss_dict = {"loss": 0.0, "loss_ce": 0.0, "loss_mse": 0.0}
        for p in logits:
            loss_dict['loss_ce'] += self.ce(rearrange(p, "b n_classes seq_len -> (b seq_len) n_classes"),
                                            rearrange(targets, "b seq_len -> (b seq_len)"))

            loss_dict['loss_mse'] += torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                                                     F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                                            min=0,
                                                            max=self.mse_clip_val))
        # seq_len = attn_mask.shape[1]
        # if self.cfg.MODEL.BCE_WEIGHT:
        #     weight = torch.ones(self.cfg.MODEL.NUM_CLASSES+1, seq_len)
        #     # weight = torch.ones(self.cfg.MODEL.NUM_CLASSES+1, seq_len)
        #     weight[0] = self.cfg.MODEL.BCE_WEIGHT
        #     weight = weight.cuda()
        # else:
        #     weight = None
        # self.bce = nn.BCEWithLogitsLoss(weight, reduction='mean')
        # attn_mask = attn_mask.permute(0,2,1)
        # attn_mask = attn_mask.reshape(-1, seq_len)
        # for attn in pred_attn:
        #     attn = attn.reshape(-1, seq_len)    
        #     bce = self.bce(attn, attn_mask)
        #     loss_dict['loss_bce'] += bce

        # loss_dict['loss'] = loss_dict['loss_ce'] + self.mse_fraction * loss_dict['loss_mse'] + self.mask_ce_fraction * loss_dict['loss_bce']
        loss_dict['loss'] = loss_dict['loss_ce'] + self.mse_fraction * loss_dict['loss_mse']

        return loss_dict

def get_loss_func(cfg: CfgNode):
    """
     Retrieve the loss given the loss name.
    :param cfg:
    :return:
    """
    if cfg.MODEL.ACTION_MASK:
        return CEplusMSE_SlotBCE(cfg)

    if cfg.MODEL.LOSS_FUNC == 'ce_mse':
        return CEplusMSE(cfg)
    else:
        raise NotImplementedError("Loss {} is not supported".format(cfg.LOSS.TYPE))

