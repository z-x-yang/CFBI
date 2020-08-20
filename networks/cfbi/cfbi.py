import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.loss import Concat_CrossEntropyLoss
from networks.layers.matching import global_matching, global_matching_for_eval, local_matching, foreground2background
from networks.layers.attention import calculate_attention_head, calculate_attention_head_for_eval
from networks.cfbi.ensembler import CollaborativeEnsembler, DynamicPreHead

class CFBI(nn.Module):
    def __init__(self, cfg, feature_extracter):
        super(CFBI, self).__init__()
        self.cfg = cfg
        self.epsilon = cfg.MODEL_EPSILON

        self.feature_extracter=feature_extracter

        self.seperate_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, kernel_size=3, stride=1, padding=1, groups=cfg.MODEL_ASPP_OUTDIM)
        self.bn1 = nn.GroupNorm(cfg.MODEL_GN_GROUPS, cfg.MODEL_ASPP_OUTDIM)
        self.relu1 = nn.ReLU(True)
        self.embedding_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_SEMANTIC_EMBEDDING_DIM, 1, 1)
        self.bn2 = nn.GroupNorm(cfg.MODEL_GN_EMB_GROUPS, cfg.MODEL_SEMANTIC_EMBEDDING_DIM)
        self.relu2 = nn.ReLU(True)
        self.semantic_embedding=nn.Sequential(*[self.seperate_conv, self.bn1, self.relu1, self.embedding_conv, self.bn2, self.relu2])

        self.bg_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.fg_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

        self.criterion = Concat_CrossEntropyLoss(cfg.TRAIN_TOP_K_PERCENT_PIXELS, cfg.TRAIN_HARD_MINING_STEP)

        for m in self.semantic_embedding:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        self.dynamic_seghead = CollaborativeEnsembler(
            in_dim=cfg.MODEL_SEMANTIC_EMBEDDING_DIM * 3 + cfg.MODEL_PRE_HEAD_EMBEDDING_DIM,
            attention_dim=cfg.MODEL_SEMANTIC_EMBEDDING_DIM * 4,
            embed_dim=cfg.MODEL_HEAD_EMBEDDING_DIM,
            refine_dim=cfg.MODEL_REFINE_CHANNELS,
            low_level_dim=cfg.MODEL_LOW_LEVEL_INPLANES)

        in_dim = 2 + len(cfg.MODEL_MULTI_LOCAL_DISTANCE)
        if cfg.MODEL_MATCHING_BACKGROUND:
            in_dim += 1 +len(cfg.MODEL_MULTI_LOCAL_DISTANCE)
        self.dynamic_prehead = DynamicPreHead(
            in_dim=in_dim, 
            embed_dim=cfg.MODEL_PRE_HEAD_EMBEDDING_DIM)


    def forward(self, input, ref_frame_label, previous_frame_mask, current_frame_mask,
            gt_ids, step=0, tf_board=False):
        x, low_level = self.extract_feature(input)
        ref_frame_embedding, previous_frame_embedding, current_frame_embedding = torch.split(x, split_size_or_sections=int(x.size(0)/3), dim=0)
        _, _, current_low_level = torch.split(low_level, split_size_or_sections=int(x.size(0)/3), dim=0)
        bs, c, h, w = current_frame_embedding.size()
        tmp_dic, boards = self.before_seghead_process(
            ref_frame_embedding,
            previous_frame_embedding,
            current_frame_embedding,
            ref_frame_label,
            previous_frame_mask,
            gt_ids,
            current_low_level=current_low_level,tf_board=tf_board)
        label_dic=[]
        all_pred = []
        for i in range(bs):
            tmp_pred_logits = tmp_dic[i]
            tmp_pred_logits = nn.functional.interpolate(tmp_pred_logits, size=(input.shape[2],input.shape[3]), mode='bilinear', align_corners=True)
            tmp_dic[i] = tmp_pred_logits
            label_tmp, obj_num = current_frame_mask[i], gt_ids[i]
            label_dic.append(label_tmp.long())
            pred = tmp_pred_logits
            preds_s = torch.argmax(pred,dim=1)
            all_pred.append(preds_s)
        all_pred = torch.cat(all_pred, dim=0)

        return self.criterion(tmp_dic, label_dic, step), all_pred, boards

    def forward_for_eval(self, ref_embeddings, ref_masks, prev_embedding, prev_mask, current_frame, pred_size, gt_ids):
        current_frame_embedding, current_low_level = self.extract_feature(current_frame)
        if prev_embedding is None:
            return None, current_frame_embedding
        else:
            bs,c,h,w = current_frame_embedding.size()
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
                pred = nn.functional.interpolate(pred, size=(pred_size[0],pred_size[1]), mode='bilinear',align_corners=True)
                all_pred.append(pred)
            all_pred = torch.cat(all_pred, dim=0)
            all_pred = torch.softmax(all_pred, dim=1)
            return all_pred, current_frame_embedding

    def extract_feature(self, x):
        x, low_level=self.feature_extracter(x)
        x = self.semantic_embedding(x)
        return x, low_level

    def before_seghead_process(self,
            ref_frame_embedding=None, previous_frame_embedding=None, current_frame_embedding=None,
            ref_frame_label=None, previous_frame_mask=None,
            gt_ids=None, current_low_level=None, tf_board=False):

        cfg = self.cfg
        
        dic_tmp=[]
        bs,c,h,w = current_frame_embedding.size()

        if self.training:
            scale_ref_frame_label = torch.nn.functional.interpolate(ref_frame_label.float(),size=(h,w),mode='nearest')
            scale_ref_frame_label = scale_ref_frame_label.int()
        else:
            scale_ref_frame_labels = []
            for each_ref_frame_label in ref_frame_label:
                each_scale_ref_frame_label = torch.nn.functional.interpolate(each_ref_frame_label.float(),size=(h,w),mode='nearest')
                each_scale_ref_frame_label = each_scale_ref_frame_label.int()
                scale_ref_frame_labels.append(each_scale_ref_frame_label)

        scale_previous_frame_label=torch.nn.functional.interpolate(previous_frame_mask.float(),size=(h,w),mode='nearest')
        scale_previous_frame_label=scale_previous_frame_label.int()

        boards = {'image': {}, 'scalar': {}}

        for n in range(bs):
            ref_obj_ids = torch.arange(0, gt_ids[n] + 1, device=current_frame_embedding.device).int().view(-1, 1, 1, 1)
            obj_num = ref_obj_ids.size(0)
            if gt_ids[n] > 0:
                dis_bias = torch.cat([self.bg_bias, self.fg_bias.expand(gt_ids[n], -1, -1, -1)], dim=0)
            else:
                dis_bias = self.bg_bias

            seq_current_frame_embedding = current_frame_embedding[n]
            seq_current_frame_embedding = seq_current_frame_embedding.permute(1,2,0)
            ########################Global dist map
            
            if self.training:
                seq_ref_frame_embedding = ref_frame_embedding[n]
                seq_ref_frame_embedding = seq_ref_frame_embedding.permute(1,2,0)

                seq_ref_frame_label = (scale_ref_frame_label[n].int() == ref_obj_ids).float()
                to_cat_ref_frame = seq_ref_frame_label
                seq_ref_frame_label = seq_ref_frame_label.squeeze(1).permute(1,2,0)

                global_matching_fg = global_matching(
                    reference_embeddings=seq_ref_frame_embedding, 
                    query_embeddings=seq_current_frame_embedding, 
                    reference_labels=seq_ref_frame_label, 
                    n_chunks=cfg.TRAIN_GLOBAL_CHUNKS, 
                    dis_bias=dis_bias, 
                    atrous_rate=cfg.TRAIN_GLOBAL_ATROUS_RATE,
                    use_float16=cfg.MODEL_FLOAT16_MATCHING)
            else:
                all_reference_embeddings = []
                all_reference_labels = []
                seq_ref_frame_labels = []
                for idx in range(len(scale_ref_frame_labels)):
                    each_ref_frame_embedding = ref_frame_embedding[idx]
                    scale_ref_frame_label = scale_ref_frame_labels[idx]

                    seq_ref_frame_embedding = each_ref_frame_embedding[n]
                    seq_ref_frame_embedding = seq_ref_frame_embedding.permute(1,2,0)
                    all_reference_embeddings.append(seq_ref_frame_embedding)

                    seq_ref_frame_label = (scale_ref_frame_label[n].int() == ref_obj_ids).float()
                    seq_ref_frame_labels.append(seq_ref_frame_label)
                    seq_ref_frame_label = seq_ref_frame_label.squeeze(1).permute(1,2,0)
                    all_reference_labels.append(seq_ref_frame_label)
                global_matching_fg = global_matching_for_eval(
                    all_reference_embeddings=all_reference_embeddings,
                    query_embeddings=seq_current_frame_embedding, 
                    all_reference_labels=all_reference_labels,
                    n_chunks=cfg.TEST_GLOBAL_CHUNKS,
                    dis_bias=dis_bias, 
                    atrous_rate=cfg.TEST_GLOBAL_ATROUS_RATE,
                    use_float16=cfg.MODEL_FLOAT16_MATCHING)
    
    
            #########################Local dist map
            seq_prev_frame_embedding = previous_frame_embedding[n]
            seq_prev_frame_embedding = seq_prev_frame_embedding.permute(1,2,0)
            seq_previous_frame_label = (scale_previous_frame_label[n].int() == ref_obj_ids).float()
            to_cat_previous_frame = seq_previous_frame_label
            seq_previous_frame_label = seq_previous_frame_label.squeeze(1).permute(1,2,0)
            local_matching_fg = local_matching(
                prev_frame_embedding=seq_prev_frame_embedding, 
                query_embedding=seq_current_frame_embedding, 
                prev_frame_labels=seq_previous_frame_label,
                multi_local_distance=cfg.MODEL_MULTI_LOCAL_DISTANCE,
                dis_bias=dis_bias, 
                use_float16=cfg.MODEL_FLOAT16_MATCHING, 
                atrous_rate=cfg.TRAIN_LOCAL_ATROUS_RATE if self.training else cfg.TEST_LOCAL_ATROUS_RATE,
                allow_downsample=cfg.MODEL_LOCAL_DOWNSAMPLE,
                allow_parallel=cfg.TRAIN_LOCAL_PARALLEL if self.training else cfg.TEST_LOCAL_PARALLEL)
    
            ##########################
    
            to_cat_current_frame_embedding = current_frame_embedding[n].unsqueeze(0).expand((obj_num,-1,-1,-1))
            to_cat_prev_frame_embedding = previous_frame_embedding[n].unsqueeze(0).expand((obj_num,-1,-1,-1))
    
            to_cat_global_matching_fg = global_matching_fg.squeeze(0).permute(2,3,0,1)
            to_cat_local_matching_fg = local_matching_fg.squeeze(0).permute(2,3,0,1)

            if cfg.MODEL_MATCHING_BACKGROUND:
                to_cat_global_matching_bg = foreground2background(to_cat_global_matching_fg, gt_ids[n] + 1)
                reshaped_prev_nn_feature_n = to_cat_local_matching_fg.permute(0, 2, 3, 1).unsqueeze(1)
                to_cat_local_matching_bg = foreground2background(reshaped_prev_nn_feature_n, gt_ids[n] + 1)
                to_cat_local_matching_bg = to_cat_local_matching_bg.permute(0, 4, 2, 3, 1).squeeze(-1)

            pre_to_cat = torch.cat((to_cat_global_matching_fg, to_cat_local_matching_fg, to_cat_previous_frame), 1)

            if cfg.MODEL_MATCHING_BACKGROUND:
                pre_to_cat = torch.cat([pre_to_cat, to_cat_local_matching_bg, to_cat_global_matching_bg], 1)

            pre_to_cat = self.dynamic_prehead(pre_to_cat)

            to_cat = torch.cat((to_cat_current_frame_embedding, to_cat_prev_frame_embedding * to_cat_previous_frame, to_cat_prev_frame_embedding * (1 - to_cat_previous_frame), pre_to_cat),1)

            if self.training:
                attention_head = calculate_attention_head(
                    ref_frame_embedding[n].unsqueeze(0).expand((obj_num,-1,-1,-1)),
                    to_cat_ref_frame, 
                    previous_frame_embedding[n].unsqueeze(0).expand((obj_num,-1,-1,-1)), 
                    to_cat_previous_frame,
                    epsilon=self.epsilon)
            else:
                attention_head = calculate_attention_head_for_eval(
                    ref_frame_embedding,
                    seq_ref_frame_labels,
                    previous_frame_embedding[n].unsqueeze(0).expand((obj_num,-1,-1,-1)), 
                    to_cat_previous_frame, 
                    epsilon=self.epsilon)

            low_level_feat = current_low_level[n].unsqueeze(0)

            pred = self.dynamic_seghead(to_cat, attention_head, low_level_feat)

            dic_tmp.append(pred)

        return dic_tmp, boards

def get_module():
    return CFBI
