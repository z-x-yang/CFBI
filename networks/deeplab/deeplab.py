import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.deeplab.aspp import build_aspp
from networks.deeplab.decoder import build_decoder
from networks.deeplab.backbone import build_backbone
from networks.layers.normalization import FrozenBatchNorm2d

class DeepLab(nn.Module):
    def __init__(self, 
            backbone='resnet', 
            output_stride=16,
            freeze_bn=True):
        super(DeepLab, self).__init__()

        if freeze_bn == True:
            print("Use frozen BN in DeepLab!")
            BatchNorm = FrozenBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(backbone, BatchNorm)


    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)

        return x, low_level_feat


    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    input = torch.rand(2, 3, 513, 513)
    output = model(input)
    print(output.size())


