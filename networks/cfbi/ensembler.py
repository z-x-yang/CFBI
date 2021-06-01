import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.attention import IA_gate
from networks.layers.gct import Bottleneck, GCT
from networks.layers.aspp import ASPP

class CollaborativeEnsembler(nn.Module):
    def __init__(self,
            in_dim=256,
            attention_dim=400,
            embed_dim=100,
            refine_dim=48,
            low_level_dim=256):
        super(CollaborativeEnsembler, self).__init__()
        self.embed_dim = embed_dim
        IA_in_dim = attention_dim

        # stage 1
        self.IA1 = IA_gate(IA_in_dim, in_dim)
        self.layer1 = Bottleneck(in_dim, embed_dim)

        self.IA2 = IA_gate(IA_in_dim, embed_dim)
        self.layer2 = Bottleneck(embed_dim, embed_dim, 1, 2)

        # stage2
        self.IA3 = IA_gate(IA_in_dim, embed_dim)
        self.layer3 = Bottleneck(embed_dim, embed_dim * 2, 2)

        self.IA4 = IA_gate(IA_in_dim, embed_dim * 2)
        self.layer4 = Bottleneck(embed_dim * 2, embed_dim * 2, 1, 2)

        self.IA5 = IA_gate(IA_in_dim, embed_dim * 2)
        self.layer5 = Bottleneck(embed_dim * 2, embed_dim * 2, 1, 4)

        # stage3
        self.IA6 = IA_gate(IA_in_dim, embed_dim * 2)
        self.layer6 = Bottleneck(embed_dim * 2, embed_dim * 2, 2)

        self.IA7 = IA_gate(IA_in_dim, embed_dim * 2)
        self.layer7 = Bottleneck(embed_dim * 2, embed_dim * 2, 1, 2)

        self.IA8 = IA_gate(IA_in_dim, embed_dim * 2)
        self.layer8 = Bottleneck(embed_dim * 2, embed_dim * 2, 1, 4)

        # ASPP
        self.IA9 = IA_gate(IA_in_dim, embed_dim * 2)
        self.ASPP = ASPP()

        # Decoder
        self.GCT_sc = GCT(low_level_dim + embed_dim)
        self.conv_sc = nn.Conv2d(low_level_dim + embed_dim, refine_dim, 1, bias=False)
        self.bn_sc = nn.GroupNorm(int(refine_dim / 4), refine_dim)
        self.relu = nn.ReLU(inplace=True)


        self.IA10 = IA_gate(IA_in_dim, embed_dim + refine_dim)
        self.conv1 = nn.Conv2d(embed_dim + refine_dim, int(embed_dim / 2), kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, int(embed_dim / 2))


        self.IA11 = IA_gate(IA_in_dim, int(embed_dim / 2))
        self.conv2 = nn.Conv2d(int(embed_dim / 2), int(embed_dim / 2), kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, int(embed_dim / 2))

        # Output
        self.IA_final_fg = nn.Linear(IA_in_dim, int(embed_dim / 2) + 1)
        self.IA_final_bg = nn.Linear(IA_in_dim, int(embed_dim / 2) + 1)

        nn.init.kaiming_normal_(self.conv_sc.weight,mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight,mode='fan_out', nonlinearity='relu')

        
################TODO
    def forward(self, x, IA_head=None, low_level_feat=None):
        # stage 1
        x = self.IA1(x, IA_head)
        x = self.layer1(x)

        x = self.IA2(x, IA_head)
        x = self.layer2(x)

        low_level_feat = torch.cat([low_level_feat.expand(x.size()[0], -1, -1, -1), x], dim=1)

        # stage 2
        x = self.IA3(x, IA_head)
        x = self.layer3(x)

        x = self.IA4(x, IA_head)
        x = self.layer4(x)

        x = self.IA5(x, IA_head)
        x = self.layer5(x)

        # stage 3
        x = self.IA6(x, IA_head)
        x = self.layer6(x)

        x = self.IA7(x, IA_head)
        x = self.layer7(x)

        x = self.IA8(x, IA_head)
        x = self.layer8(x)

        x = self.IA9(x, IA_head)
        x = self.ASPP(x)

        x = self.decoder(x, low_level_feat, IA_head)

        fg_logit = self.IA_logit(x, IA_head, self.IA_final_fg)
        bg_logit = self.IA_logit(x, IA_head, self.IA_final_bg)

        pred = self.augment_background_logit(fg_logit, bg_logit)

        return pred

    def IA_logit(self, x, IA_head, IA_final):
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        IA_output = IA_final(IA_head)
        IA_weight = IA_output[:, :c]
        IA_bias = IA_output[:, -1]
        IA_weight = IA_weight.view(n, c, 1, 1)
        IA_bias = IA_bias.view(-1)
        logit = F.conv2d(x, weight=IA_weight, bias=IA_bias, groups=n).view(n, 1, h, w)
        return logit

    def decoder(self, x, low_level_feat, IA_head):
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bicubic', align_corners=True)

        low_level_feat = self.GCT_sc(low_level_feat)
        low_level_feat = self.conv_sc(low_level_feat)
        low_level_feat = self.bn_sc(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = torch.cat([x, low_level_feat], dim=1)
        x = self.IA10(x, IA_head)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.IA11(x, IA_head)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def augment_background_logit(self, fg_logit, bg_logit):
        #  We augment the logit of absolute background by using the relative background logit of all the 
        #  foreground objects.
        obj_num = fg_logit.size(0)
        pred = fg_logit
        if obj_num > 1:
            bg_logit = bg_logit[1:obj_num, :, :, :]
            aug_bg_logit, _ = torch.min(bg_logit, dim=0, keepdim=True)
            pad = torch.zeros(aug_bg_logit.size(), device=aug_bg_logit.device).expand(obj_num - 1, -1, -1, -1)
            aug_bg_logit = torch.cat([aug_bg_logit, pad], dim=0)
            pred = pred + aug_bg_logit
        pred = pred.permute(1,0,2,3)
        return pred

class DynamicPreHead(nn.Module):
    def __init__(self, in_dim=3, embed_dim=100, kernel_size=1):
        super(DynamicPreHead,self).__init__()
        self.conv=nn.Conv2d(in_dim,embed_dim,kernel_size=kernel_size,stride=1,padding=int((kernel_size-1)/2))
        self.bn = nn.GroupNorm(int(embed_dim / 4), embed_dim)
        self.relu = nn.ReLU(True)
        nn.init.kaiming_normal_(self.conv.weight,mode='fan_out',nonlinearity='relu')

################TODO
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CollaborativeEnsemblerMS(nn.Module):
    def __init__(self,
            in_dim_4x, 
            in_dim_8x, 
            in_dim_16x,
            attention_dim=400,
            embed_dim=100,
            refine_dim=48,
            low_level_dim=256):
        super(CollaborativeEnsemblerMS, self).__init__()
        IA_in_dim = attention_dim

        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.S1_IA1 = IA_gate(IA_in_dim, in_dim_4x)
        self.S1_layer1 = Bottleneck(in_dim_4x, embed_dim)

        self.S1_IA2 = IA_gate(IA_in_dim, embed_dim)
        self.S1_layer2 = Bottleneck(embed_dim, embed_dim, 1, 2)

        # stage2
        self.S2_IA1 = IA_gate(IA_in_dim, embed_dim)
        self.S2_layer1 = Bottleneck(embed_dim, embed_dim * 2, 2)

        self.S2_IA2 = IA_gate(IA_in_dim, embed_dim * 2 + in_dim_8x)
        self.S2_layer2 = Bottleneck(embed_dim * 2 + in_dim_8x, embed_dim * 2, 1, 2)

        self.S2_IA3 = IA_gate(IA_in_dim, embed_dim * 2)
        self.S2_layer3 = Bottleneck(embed_dim * 2, embed_dim * 2, 1, 4)


        # stage3
        self.S3_IA1 = IA_gate(IA_in_dim, embed_dim * 2)
        self.S3_layer1 = Bottleneck(embed_dim * 2, embed_dim * 2, 2)

        self.S3_IA2 = IA_gate(IA_in_dim, embed_dim * 2 + in_dim_16x)
        self.S3_layer2 = Bottleneck(embed_dim * 2 + in_dim_16x, embed_dim * 2, 1, 2)

        self.S3_IA3 = IA_gate(IA_in_dim, embed_dim * 2)
        self.S3_layer3 = Bottleneck(embed_dim * 2, embed_dim * 2, 1, 4)


        self.ASPP_IA = IA_gate(IA_in_dim, embed_dim * 2)
        self.ASPP = ASPP()

        # Decoder
        self.GCT_sc = GCT(low_level_dim + embed_dim)
        self.conv_sc = nn.Conv2d(low_level_dim + embed_dim, refine_dim, 1, bias=False)
        self.bn_sc = nn.GroupNorm(int(refine_dim / 4), refine_dim)
        self.relu = nn.ReLU(inplace=True)


        self.IA10 = IA_gate(IA_in_dim, embed_dim + refine_dim)
        self.conv1 = nn.Conv2d(embed_dim + refine_dim, int(embed_dim / 2), kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, int(embed_dim / 2))


        self.IA11 = IA_gate(IA_in_dim, int(embed_dim / 2))
        self.conv2 = nn.Conv2d(int(embed_dim / 2), int(embed_dim / 2), kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, int(embed_dim / 2))

        # Output
        self.IA_final_fg = nn.Linear(IA_in_dim, int(embed_dim / 2) + 1)
        self.IA_final_bg = nn.Linear(IA_in_dim, int(embed_dim / 2) + 1)

        nn.init.kaiming_normal_(self.conv_sc.weight,mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight,mode='fan_out', nonlinearity='relu')


################TODO
    def forward(self, all_x, all_IA_head=None, low_level_feat=None):
        x_4x, x_8x, x_16x = all_x
        IA_head = all_IA_head[0]

        # stage 1
        x = self.S1_IA1(x_4x, IA_head)
        x = self.S1_layer1(x)

        x = self.S1_IA2(x, IA_head)
        x = self.S1_layer2(x)

        low_level_feat = torch.cat([low_level_feat.expand(x.size()[0], -1, -1, -1), x], dim=1)

        # stage 2
        x = self.S2_IA1(x, IA_head)
        x = self.S2_layer1(x)

        x = torch.cat([x, x_8x], dim=1)
        x = self.S2_IA2(x, IA_head)
        x = self.S2_layer2(x)

        x = self.S2_IA3(x, IA_head)
        x = self.S2_layer3(x)

        # stage 3
        x = self.S3_IA1(x, IA_head)
        x = self.S3_layer1(x)

        x = torch.cat([x, x_16x], dim=1)
        x = self.S3_IA2(x, IA_head)
        x = self.S3_layer2(x)

        x = self.S3_IA3(x, IA_head)
        x = self.S3_layer3(x)

        # ASPP + Decoder
        x = self.ASPP_IA(x, IA_head)
        x = self.ASPP(x)

        x = self.decoder(x, low_level_feat, IA_head)

        fg_logit = self.IA_logit(x, IA_head, self.IA_final_fg)
        bg_logit = self.IA_logit(x, IA_head, self.IA_final_bg)

        pred = self.augment_background_logit(fg_logit, bg_logit)

        return pred

    def IA_logit(self, x, IA_head, IA_final):
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        IA_output = IA_final(IA_head)
        IA_weight = IA_output[:, :c]
        IA_bias = IA_output[:, -1]
        IA_weight = IA_weight.view(n, c, 1, 1)
        IA_bias = IA_bias.view(-1)
        logit = F.conv2d(x, weight=IA_weight, bias=IA_bias, groups=n).view(n, 1, h, w)
        return logit

    def decoder(self, x, low_level_feat, IA_head):
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bicubic', align_corners=True)

        low_level_feat = self.GCT_sc(low_level_feat)
        low_level_feat = self.conv_sc(low_level_feat)
        low_level_feat = self.bn_sc(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = torch.cat([x, low_level_feat], dim=1)
        x = self.IA10(x, IA_head)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.IA11(x, IA_head)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def augment_background_logit(self, fg_logit, bg_logit):
        #  We augment the logit of absolute background by using the relative background logit of all the 
        #  foreground objects.
        obj_num = fg_logit.size(0)
        pred = fg_logit
        if obj_num > 1:
            bg_logit = bg_logit[1:obj_num, :, :, :]
            aug_bg_logit, _ = torch.min(bg_logit, dim=0, keepdim=True)
            pad = torch.zeros(aug_bg_logit.size(), device=aug_bg_logit.device).expand(obj_num - 1, -1, -1, -1)
            aug_bg_logit = torch.cat([aug_bg_logit, pad], dim=0)
            pred = pred + aug_bg_logit
        pred = pred.permute(1,0,2,3)
        return pred
