import torch
import torch.nn as nn
from torchvision import models


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG = models.vgg19(pretrained=True).features

        for parameter in self.VGG.parameters():
            parameter.requires_grad_(False)

    def norm(self, image):
        DEV = image.device
        mean = torch.tensor([0.485, 0.456, 0.406]).to(DEV)
        std = torch.tensor([0.229, 0.224, 0.225]).to(DEV)
        mean = mean.view(1,-1,1,1)
        std = std.view(1,-1,1,1)
        im_norm = (image - mean) / std
        
        return im_norm

    def get_features(self, image, norm=True):
        image_re = image.repeat(1,3,1,1) if image.shape[1] == 1 else image
        image_re = self.norm(image_re) if norm else image_re

        vgg_convs = {'0': 'conv1_1',
                     '5': 'conv2_1',
                     '10': 'conv3_1',
                     '19': 'conv4_1',
                     '21': 'conv4_2',
                     '28': 'conv5_1',
                     '31': 'conv5_2'}  
        
        features = {}
        x = image_re
        for name, layer in self.VGG._modules.items():
            x = layer(x)   
            if name in vgg_convs:
                features[vgg_convs[name]] = x
        
        return features

    def im_loss(self, target, content, norm=True):
        content_features = self.get_features(content, norm)
        target_features = self.get_features(target, norm)

        content_loss = 0
        content_loss += torch.mean((target_features['conv2_1'] - content_features['conv2_1']) ** 2).mean()
        content_loss += torch.mean((target_features['conv3_1'] - content_features['conv3_1']) ** 2).mean()

        return content_loss
    
    def feature_loss(self, target_feature, content1_feature, content2_feature):
        content_loss = 0
        content_loss += torch.mean((target_feature - content1_feature) ** 2).mean()
        content_loss += torch.mean((target_feature - content2_feature) ** 2).mean()

        return content_loss

