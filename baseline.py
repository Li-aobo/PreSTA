import torch
import torch.nn as nn
import timm

SUPPORTED_BACKBONES = ('swin_base_patch4_window7_224', 'resnet50')


class Baseline(nn.Module):

    def __init__(self, backbone_name, pretrain=True, output_f=False):
        super(Baseline, self).__init__()
        self.output_f = output_f
        self.backbone_name = backbone_name

        self.backbone, in_features = self.get_backbone(backbone_name, pretrain)

        self.regressor = nn.Linear(in_features=in_features, out_features=1)

        self.regressor.apply(self.weight_init)

    def forward(self, x):
        x = self.backbone(x)
        f = x.reshape(x.size(0), -1)

        x = self.regressor(f)

        if self.output_f:
            return x, f
        return x

    def get_backbone(self, name, pretrain=True):
        if name not in SUPPORTED_BACKBONES:
            raise ValueError('Unsupported backbone: {}. Supported backbones: {}'.format(
                name, ', '.join(SUPPORTED_BACKBONES)))

        try:
            timm_model = timm.create_model(name, pretrained=pretrain, num_classes=0)
            in_features = timm_model.num_features

            return timm_model, in_features

        except Exception as e:
            print(f"Failed to create timm model '{name}'.")
            raise e

    def weight_init(self, net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class HierarchicalBackbone(nn.Module):

    def __init__(self, backbone_name='resnet50', pretrain=True, output_f=False,
                 out_indices=(0, 1, 2, 3)):
        super(HierarchicalBackbone, self).__init__()
        self.output_f = output_f
        self.backbone_name = backbone_name
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError('Unsupported backbone: {}. Supported backbones: {}'.format(
                backbone_name, ', '.join(SUPPORTED_BACKBONES)))

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrain,
            features_only=True,
            out_indices=out_indices
        )
        self.feature_channels = self.backbone.feature_info.channels()
        self.feature_dim = sum(self.feature_channels)

        self.regressor = nn.Linear(self.feature_dim, 1)
        self.regressor.apply(self.weight_init)

    def forward(self, x):
        feature_maps = self.backbone(x)
        pooled_features = [
            self._global_avg_pool(feature, channels)
            for feature, channels in zip(feature_maps, self.feature_channels)
        ]
        features = torch.cat(pooled_features, dim=1)
        pred = self.regressor(features)

        if self.output_f:
            return pred, features
        return pred

    @staticmethod
    def _global_avg_pool(feature, channels):
        if feature.dim() != 4:
            return feature.reshape(feature.size(0), -1)
        if feature.shape[1] == channels:
            return feature.mean(dim=(2, 3))
        if feature.shape[-1] == channels:
            return feature.mean(dim=(1, 2))
        return feature.flatten(2).mean(dim=-1)

    @staticmethod
    def weight_init(net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
