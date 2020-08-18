import torch
import torch.nn as nn
import os

class Concat_BCEWithLogitsLoss(nn.Module):
    def __init__(self, top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(Concat_BCEWithLogitsLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert(top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step
        if self.top_k_percent_pixels == None:
            self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.bceloss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, dic_tmp, y, step):
        total_loss = []
        for i in range(len(dic_tmp)):
            pred_logits = dic_tmp[i]
            gts = y[i]
            if self.top_k_percent_pixels == None:
                final_loss = self.bceloss(pred_logits, gts)
            else:
                # Only compute the loss for top k percent pixels.
                # First, compute the loss for all pixels. Note we do not put the loss
                # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.size(2) * pred_logits.size(3))
                pred_logits = pred_logits.view(-1, pred_logits.size(
                    1), pred_logits.size(2) * pred_logits.size(3))
                gts = gts.view(-1, gts.size(1), gts.size(2) * gts.size(3))
                pixel_losses = self.bceloss(pred_logits, gts)
                if self.hard_example_mining_step == 0:
                    top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
                else:
                    ratio = min(
                        1.0, step / float(self.hard_example_mining_step))
                    top_k_pixels = int(
                        (ratio * self.top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
                _, top_k_indices = torch.topk(
                    pixel_losses, k=top_k_pixels, dim=2)

                final_loss = nn.BCEWithLogitsLoss(
                    weight=top_k_indices, reduction='mean')(pred_logits, gts)
            final_loss = final_loss.unsqueeze(0)
            total_loss.append(final_loss)
        total_loss = torch.cat(total_loss, dim=0)
        return total_loss


class Concat_CrossEntropyLoss(nn.Module):
    def __init__(self, top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(Concat_CrossEntropyLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert(top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step
        if self.top_k_percent_pixels == None:
            self.celoss = nn.CrossEntropyLoss(
                ignore_index=255, reduction='mean')
        else:
            self.celoss = nn.CrossEntropyLoss(
                ignore_index=255, reduction='none')

    def forward(self, dic_tmp, y, step):
        total_loss = []
        for i in range(len(dic_tmp)):
            pred_logits = dic_tmp[i]
            gts = y[i]
            if self.top_k_percent_pixels == None:
                final_loss = self.celoss(pred_logits, gts)
            else:
                # Only compute the loss for top k percent pixels.
                # First, compute the loss for all pixels. Note we do not put the loss
                # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.size(2) * pred_logits.size(3))
                pred_logits = pred_logits.view(-1, pred_logits.size(
                    1), pred_logits.size(2) * pred_logits.size(3))
                gts = gts.view(-1, gts.size(1) * gts.size(2))
                pixel_losses = self.celoss(pred_logits, gts)
                if self.hard_example_mining_step == 0:
                    top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
                else:
                    ratio = min(
                        1.0, step / float(self.hard_example_mining_step))
                    top_k_pixels = int(
                        (ratio * self.top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
                top_k_loss, top_k_indices = torch.topk(
                    pixel_losses, k=top_k_pixels, dim=1)

                final_loss = torch.mean(top_k_loss)
            final_loss = final_loss.unsqueeze(0)
            total_loss.append(final_loss)
        total_loss = torch.cat(total_loss, dim=0)
        return total_loss
