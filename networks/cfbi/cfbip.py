import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.loss import Concat_CrossEntropyLoss
from networks.layers.matching import global_matching, global_matching_for_eval, local_matching, foreground2background
from networks.layers.attention import calculate_attention_head, calculate_attention_head_for_eval
from networks.layers.fpn import FPN
from networks.cfbi.ensembler import CollaborativeEnsemblerMS

class CFBIP(nn.Module):
    def __init__(self, cfg, feature_extracter):
        super(CFBIP, self).__init__()
        self.cfg = cfg
        self.epsilon = cfg.MODEL_EPSILON

        self.feature_extracter=feature_extracter

        self.fpn = FPN(in_dim_4x=cfg.MODEL_ASPP_OUTDIM, in_dim_8x=512, in_dim_16x=cfg.MODEL_ASPP_OUTDIM, out_dim=cfg.MODEL_SEMANTIC_EMBEDDING_DIM)

        self.bg_bias = nn.Parameter(torch.zeros(3, 1, 1, 1))
        self.fg_bias = nn.Parameter(torch.zeros(3, 1, 1, 1))

        self.criterion = Concat_CrossEntropyLoss(cfg.TRAIN_TOP_K_PERCENT_PIXELS, cfg.TRAIN_HARD_MINING_STEP)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.dynamic_seghead = CollaborativeEnsemblerMS(
            in_dim_4x=cfg.MODEL_SEMANTIC_EMBEDDING_DIM * 3 + 3 + 2 * len(cfg.MODEL_MULTI_LOCAL_DISTANCE[0]), 
            in_dim_8x=cfg.MODEL_SEMANTIC_EMBEDDING_DIM * 3 + 3 + 2 * len(cfg.MODEL_MULTI_LOCAL_DISTANCE[1]), 
            in_dim_16x=cfg.MODEL_SEMANTIC_EMBEDDING_DIM * 3 + 3 + 2 * len(cfg.MODEL_MULTI_LOCAL_DISTANCE[2]),
            attention_dim=cfg.MODEL_SEMANTIC_EMBEDDING_DIM * 4,
            embed_dim=cfg.MODEL_HEAD_EMBEDDING_DIM,
            refine_dim=cfg.MODEL_REFINE_CHANNELS,
            low_level_dim=cfg.MODEL_LOW_LEVEL_INPLANES)


    def forward(self, input, ref_frame_label, previous_frame_mask, current_frame_mask,
            gt_ids, step=0, tf_board=False):

        return None, None, None

    def forward_for_eval(self, ref_embeddings, ref_masks, prev_embedding, prev_mask, current_frame,
                         pred_size, gt_ids, is_flipped=False):

        current_frame_embedding_4x, current_frame_embedding_8x, current_frame_embedding_16x, current_low_level = self.extract_feature(current_frame)
        current_frame_embedding = [current_frame_embedding_4x, current_frame_embedding_8x, current_frame_embedding_16x]
        
        if prev_embedding is None:
            return None, current_frame_embedding
        else:
            bs,c,h,w = current_frame_embedding_4x.size()
            tmp_dic, _ = self.before_seghead_process(
                ref_embeddings, 
                prev_embedding,
                current_frame_embedding, 
                ref_masks,
                prev_mask,
                gt_ids, 
                current_low_level=current_low_level,
                tf_board=False)
            all_pred = []
            for i in range(bs):
                pred = tmp_dic[i]

                pred = nn.functional.interpolate(pred, size=(pred_size[0], pred_size[1]),
                                                 mode='bilinear', align_corners=True)
                
                all_pred.append(pred)
            all_pred = torch.cat(all_pred, dim=0)
            all_pred = torch.softmax(all_pred, dim=1)
            return all_pred, current_frame_embedding

    def extract_feature(self, x):
        x, aspp_x, low_level, mid_level = self.feature_extracter(x, True)

        x_4x, x_8x, x_16x = self.fpn(x, mid_level, aspp_x)

        return x_4x, x_8x, x_16x, low_level

    def before_seghead_process(self,
            ref_frame_embeddings=None, previous_frame_embeddings=None, current_frame_embeddings=None,
            ref_frame_labels=None, previous_frame_mask=None,
            gt_ids=None, current_low_level=None, tf_board=False):

        cfg = self.cfg
        
        dic_tmp=[]
        boards = {}
        scale_ref_frame_labels = []
        scale_previous_frame_labels = []
        for current_frame_embedding in current_frame_embeddings:
            bs,c,h,w = current_frame_embedding.size()
            if self.training:
                scale_ref_frame_label = torch.nn.functional.interpolate(ref_frame_labels.float(), size=(h,w), mode='nearest').int()
                scale_ref_frame_labels.append(scale_ref_frame_label)
            else:
                ref_frame_embeddings = list(zip(*ref_frame_embeddings))
                all_scale_ref_frame_label = []
                for ref_frame_label in ref_frame_labels:
                    scale_ref_frame_label = torch.nn.functional.interpolate(ref_frame_label.float(), size=(h,w), mode='nearest').int()
                    all_scale_ref_frame_label.append(scale_ref_frame_label)
                scale_ref_frame_labels.append(all_scale_ref_frame_label)
            scale_previous_frame_label = torch.nn.functional.interpolate(previous_frame_mask.float(), size=(h,w), mode='nearest').int()
            scale_previous_frame_labels.append(scale_previous_frame_label)
        
        for n in range(bs):
            ref_obj_ids = torch.arange(0, gt_ids[n] + 1, device=current_frame_embedding.device).int().view(-1, 1, 1, 1)
            obj_num = ref_obj_ids.size(0)
            low_level_feat = current_low_level[n].unsqueeze(0)
            all_CE_input = []
            all_attention_head = []
            for scale_idx, current_frame_embedding, ref_frame_embedding, previous_frame_embedding, scale_ref_frame_label, scale_previous_frame_label in zip(
                range(3), current_frame_embeddings, ref_frame_embeddings, previous_frame_embeddings, scale_ref_frame_labels, scale_previous_frame_labels):
                ########################Prepare
                
                seq_current_frame_embedding = current_frame_embedding[n]
                seq_prev_frame_embedding = previous_frame_embedding[n]
                seq_previous_frame_label = (scale_previous_frame_label[n].int() == ref_obj_ids).float()

                if gt_ids[n] > 0:
                    dis_bias = torch.cat([self.bg_bias[scale_idx].unsqueeze(0), self.fg_bias[scale_idx].unsqueeze(0).expand(gt_ids[n], -1, -1, -1)], dim=0)
                else:
                    dis_bias = self.bg_bias[scale_idx].unsqueeze(0)

                ########################Global FG map
                matching_dim = cfg.MODEL_SEMANTIC_MATCHING_DIM[scale_idx]
                seq_current_frame_embedding_for_matching = seq_current_frame_embedding[:matching_dim].permute(1,2,0)
                
                if self.training:
                    seq_ref_frame_embedding = ref_frame_embedding[n]
                    seq_ref_frame_label = (scale_ref_frame_label[n].int() == ref_obj_ids).float()
                    seq_ref_frame_embedding_for_matching = seq_ref_frame_embedding[:matching_dim].permute(1,2,0)
                    seq_ref_frame_label_for_matching = seq_ref_frame_label.squeeze(1).permute(1,2,0)
                    global_matching_fg = global_matching(
                        reference_embeddings=seq_ref_frame_embedding_for_matching, 
                        query_embeddings=seq_current_frame_embedding_for_matching, 
                        reference_labels=seq_ref_frame_label_for_matching, 
                        n_chunks=cfg.TRAIN_GLOBAL_MATCHING_CHUNK[scale_idx],
                        dis_bias=dis_bias,
                        atrous_rate=cfg.TRAIN_GLOBAL_ATROUS_RATE[scale_idx],
                        use_float16=cfg.MODEL_FLOAT16_MATCHING)
                else:
                    all_scale_ref_frame_label = scale_ref_frame_label
                    all_ref_frame_embedding = ref_frame_embedding
                    all_reference_embeddings = []
                    all_reference_labels = []
                    seq_ref_frame_labels = []
                    for idx in range(len(all_scale_ref_frame_label)):
                        ref_frame_embedding = all_ref_frame_embedding[idx]
                        scale_ref_frame_label = all_scale_ref_frame_label[idx]

                        seq_ref_frame_embedding = ref_frame_embedding[n]
                        seq_ref_frame_embedding = seq_ref_frame_embedding.permute(1,2,0)

                        seq_ref_frame_label = (scale_ref_frame_label[n].int() == ref_obj_ids).float()
                        seq_ref_frame_labels.append(seq_ref_frame_label)
                        seq_ref_frame_label = seq_ref_frame_label.squeeze(1).permute(1,2,0)

                        all_reference_embeddings.append(seq_ref_frame_embedding[:, :, :matching_dim])
                        all_reference_labels.append(seq_ref_frame_label)

                    global_matching_fg = global_matching_for_eval(
                        all_reference_embeddings=all_reference_embeddings, 
                        query_embeddings=seq_current_frame_embedding_for_matching, 
                        all_reference_labels=all_reference_labels, 
                        n_chunks=cfg.TEST_GLOBAL_MATCHING_CHUNK[scale_idx],
                        dis_bias=dis_bias,
                        atrous_rate=cfg.TEST_GLOBAL_ATROUS_RATE[scale_idx],
                        use_float16=cfg.MODEL_FLOAT16_MATCHING,
                        atrous_obj_pixel_num=cfg.TEST_GLOBAL_MATCHING_MIN_PIXEL)

        
                #########################Local FG map
                seq_prev_frame_embedding_for_matching = seq_prev_frame_embedding[:matching_dim].permute(1,2,0)
                seq_previous_frame_label_for_matching = seq_previous_frame_label.squeeze(1).permute(1,2,0)
                local_matching_fg = local_matching(
                    prev_frame_embedding=seq_prev_frame_embedding_for_matching, 
                    query_embedding=seq_current_frame_embedding_for_matching, 
                    prev_frame_labels=seq_previous_frame_label_for_matching,
                    multi_local_distance=cfg.MODEL_MULTI_LOCAL_DISTANCE[scale_idx], 
                    dis_bias=dis_bias,
                    atrous_rate=cfg.TRAIN_LOCAL_ATROUS_RATE[scale_idx] if self.training else cfg.TEST_LOCAL_ATROUS_RATE[scale_idx],
                    use_float16=cfg.MODEL_FLOAT16_MATCHING,
                    allow_downsample=False,
                    allow_parallel=cfg.TRAIN_LOCAL_PARALLEL if self.training else cfg.TEST_LOCAL_PARALLEL)

                
                
                ##########################Aggregate Pixel-level Matching
                to_cat_global_matching_fg = global_matching_fg.squeeze(0).permute(2,3,0,1)
                to_cat_local_matching_fg = local_matching_fg.squeeze(0).permute(2,3,0,1)
                all_to_cat = [to_cat_global_matching_fg, to_cat_local_matching_fg, seq_previous_frame_label]

                #########################Global and Local BG map
                if cfg.MODEL_MATCHING_BACKGROUND:
                    to_cat_global_matching_bg = foreground2background(to_cat_global_matching_fg, gt_ids[n] + 1)
                    reshaped_prev_nn_feature_n = to_cat_local_matching_fg.permute(0, 2, 3, 1).unsqueeze(1)
                    to_cat_local_matching_bg = foreground2background(reshaped_prev_nn_feature_n, gt_ids[n] + 1)
                    to_cat_local_matching_bg = to_cat_local_matching_bg.permute(0, 4, 2, 3, 1).squeeze(-1)
                    all_to_cat += [to_cat_local_matching_bg, to_cat_global_matching_bg]
                
                to_cat_current_frame_embedding = current_frame_embedding[n].unsqueeze(0).expand((obj_num,-1,-1,-1))
                to_cat_prev_frame_embedding = previous_frame_embedding[n].unsqueeze(0).expand((obj_num,-1,-1,-1))
                to_cat_prev_frame_embedding_fg = to_cat_prev_frame_embedding * seq_previous_frame_label
                to_cat_prev_frame_embedding_bg = to_cat_prev_frame_embedding * (1 - seq_previous_frame_label)
                all_to_cat += [to_cat_current_frame_embedding, to_cat_prev_frame_embedding_fg, to_cat_prev_frame_embedding_bg]

                CE_input = torch.cat(all_to_cat, 1)

                ##########################Instance-level Attention

                if self.training:
                    attention_head = calculate_attention_head(
                        ref_frame_embedding[n].unsqueeze(0).expand((obj_num,-1,-1,-1)),
                        seq_ref_frame_label, 
                        previous_frame_embedding[n].unsqueeze(0).expand((obj_num,-1,-1,-1)), 
                        seq_previous_frame_label,
                        epsilon=self.epsilon)
                else:
                    attention_head = calculate_attention_head_for_eval(
                        all_ref_frame_embedding,
                        seq_ref_frame_labels, 
                        previous_frame_embedding[n].unsqueeze(0).expand((obj_num,-1,-1,-1)), 
                        seq_previous_frame_label,
                        epsilon=self.epsilon)

                all_CE_input.append(CE_input)
                all_attention_head.append(attention_head)

            ##########################Collaborative Ensembler
            pred = self.dynamic_seghead(all_CE_input, all_attention_head, low_level_feat)

            dic_tmp.append(pred)

        return dic_tmp, boards

def get_module():
    return CFBIP
