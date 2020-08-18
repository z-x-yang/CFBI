import torch
import torch.nn.functional as F
import math
from torch import nn


class IA_gate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IA_gate, self).__init__()
        self.IA = nn.Linear(in_dim, out_dim)

    def forward(self, x, IA_head):
        a = self.IA(IA_head)
        a = 1. + torch.tanh(a)
        a = a.unsqueeze(-1).unsqueeze(-1)
        x = a * x
        return x

def calculate_attention_head(ref_embedding, ref_label, prev_embedding, prev_label, epsilon=1e-5):

    ref_head = ref_embedding * ref_label
    ref_head_pos = torch.sum(ref_head, dim=(2,3))
    ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
    ref_pos_num = torch.sum(ref_label, dim=(2,3))
    ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
    ref_head_pos = ref_head_pos / (ref_pos_num + epsilon)
    ref_head_neg = ref_head_neg / (ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)

    return total_head

def calculate_attention_head_for_eval(ref_embeddings, ref_labels, prev_embedding, prev_label, epsilon=1e-5):
    total_ref_head_pos = 0.
    total_ref_head_neg = 0.
    total_ref_pos_num = 0.
    total_ref_neg_num = 0.

    for idx in range(len(ref_embeddings)):
        ref_embedding = ref_embeddings[idx]
        ref_label = ref_labels[idx]
        ref_head = ref_embedding * ref_label
        ref_head_pos = torch.sum(ref_head, dim=(2,3))
        ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
        ref_pos_num = torch.sum(ref_label, dim=(2,3))
        ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
        total_ref_head_pos = total_ref_head_pos + ref_head_pos
        total_ref_head_neg = total_ref_head_neg + ref_head_neg
        total_ref_pos_num = total_ref_pos_num + ref_pos_num
        total_ref_neg_num = total_ref_neg_num + ref_neg_num
    ref_head_pos = total_ref_head_pos / (total_ref_pos_num + epsilon)
    ref_head_neg = total_ref_head_neg / (total_ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    return total_head