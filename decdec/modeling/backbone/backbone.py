from torchvision import models
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, name='resnet50',freeze_backbone = False) -> None:
        super().__init__()
        if name =='resnet50':
            backbone = models.resnet50(pretrained = True)
            backbone = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*backbone)
        if freeze_backbone:
            self.backbone.requires_grad_ = False
        else:
            self.backbone.requires_grad_ = True
    def forward(self,x):
        # b,c,h,w
        x = self.backbone(x)
        # b,2048,h/32,w/32
        return x
