import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from math import ceil 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Slot Attention module
differences from the original repo:
1. learnable slot initializtaion
2. pad for the first frame
Inputs --> [batch_size, number_of_frame, sequence_length, hid_dim]
outputs --> slots[batch_size, number_of_frame, num_of_slots, hid_dim]; attn_masks [batch_size, number_of_frame, num_of_slots, sequence_length]
'''
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim)).to(device)
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim)).to(device)
        self.slots_sigma = self.slots_sigma.absolute()

        self.FC1 = nn.Linear(dim, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.gru = nn.GRUCell(dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        mu = self.slots_mu.expand(1, self.num_slots, -1)
        sigma = self.slots_sigma.expand(1, self.num_slots, -1)
        slots = torch.normal(mu, sigma)

        slots = slots.contiguous()
        self.register_buffer("slots", slots)
        # self.slots = nn.Embedding(num_slots, dim)
        self.slots = nn.Parameter(torch.randn(1, self.num_slots, dim)).to(device)
    def get_attention(self, slots, inputs, masks):
        slots_prev= slots
        b, n, d = inputs.shape

        inputs = self.LN(inputs)
        inputs = self.FC1(inputs)
        inputs = F.relu(inputs)
        inputs = self.FC2(inputs)*masks

        inputs = self.norm_input(inputs)   
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        dots = dots/0.2
        attn_ori = dots.softmax(dim=1) + self.eps
        # attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True) #[b, n_slot, L]
        update = torch.einsum('bjd,bij->bid', v, attn_ori)

        slots = self.gru(
            update.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )

        slots = slots.reshape(b, -1, d)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def forward(self, inputs, masks=None):
        b, n, d = inputs.shape
        # slots = self.slots.weight.expand(b,-1,-1)
        slots = self.slots.expand(b,-1,-1)
        slots, attn = self.get_attention(slots, inputs, masks)

        return slots, attn




class Decoder(nn.Module):
    def __init__(self, hid_dim, output_channel):
        super().__init__()
        self.conv1 = nn.Conv1d(hid_dim, hid_dim, 3, padding='same')
        self.conv2 = nn.Conv1d(hid_dim, hid_dim, 3, padding='same')
        self.conv3 = nn.Conv1d(hid_dim, output_channel, 3, padding='same')

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x


class SlotAttentionModule(nn.Module):
    def __init__(self, num_slots, hid_dim, out_dim=48):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        hid_dim: dimension for ConvGRU and slot attention
        """
        super().__init__()
        self.hid_dim = hid_dim

        self.num_slots = num_slots

        # self.decoder = Decoder(self.hid_dim, self.resolution, self.output_channel)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            eps = 1e-8, 
            hidden_dim = hid_dim)

        self.decoder = Decoder(hid_dim, out_dim)
        
    def forward(self, seq, masks=None):
        # `image` has shape: [batch_size, num_channels, width, height].
        bs, dim, L = seq.shape
        x = seq.permute(0, 2, 1)
        masks = masks.permute(0, 2, 1)
        x = self.fc1(x)*masks
        x = F.relu(x)
        x = self.fc2(x)*masks

        
        # Slot Attention module.
        slots, attn_masks = self.slot_attention(x, masks)
        # reshape and broadcaast attention masks 
        attn_masks = attn_masks.reshape(bs,self.num_slots, -1)
        attn_masks = attn_masks.view(attn_masks.shape[0],attn_masks.shape[1], L)
        attn_masks = attn_masks.unsqueeze(-1)

        masks = masks.permute(0,2,1)
        # recons = self.decoder(slots_combine)
        # `recons` has shape: [bs, n_frames, 3, H//4, W // 4]

        return attn_masks.view(bs, self.num_slots, L), slots

    def action_slot_forward(self, seq, masks=None):
        bs, dim, L = seq.shape
        x = seq.permute(0, 2, 1)
        masks = masks.permute(0, 2, 1)
        x = self.fc1(x)*masks
        x = F.relu(x)
        x = self.fc2(x)*masks
        
        # Slot Attention module.
        slots_ori, attn_masks = self.slot_attention(x, masks)
        # reshape and broadcaast attention masks 
        attn_masks = attn_masks.reshape(bs,self.num_slots, -1)
        attn_masks = attn_masks.view(attn_masks.shape[0],attn_masks.shape[1], L)
        attn_masks = attn_masks.unsqueeze(-1)

        slots = slots_ori.reshape(bs, self.num_slots, -1)  
        slots = slots.unsqueeze(2)

        slots_combine = slots * attn_masks
        slots_combine = slots_combine.sum(dim = 1)
        slots_combine = slots_combine.permute(0,2,1)
        slots_combine = self.decoder(slots_combine)
        # `slots_combine` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        # `recons` has shape: [bs, dim, L]
        return attn_masks.view(bs, self.num_slots, L), slots_combine

