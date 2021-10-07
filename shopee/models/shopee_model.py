import torch
import torch.nn as nn

class ShopeeModel(nn.Module):

    def __init__(
        self,
        model_name = ,
        fc_dim = 512,
        use_fc = True,
        pretrained = True,
        finetune = False,
        ):


        super(ShopeeModel,self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if model_name == 'resnext50_32x4d':
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif 'efficientnet' in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
        
        elif 'nfnet' in model_name:
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        self.finetune = finetune

        if use_fc:
            self.dropout = nn.Dropout(p=0.0)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            self.final = nn.Sequential(
                self.dropout,
                self.fc,
                self.bn,
            )
            # final_in_features = fc_dim


    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        feature = self.extract_feat(image)
        if self.use_fc:
            feature = self.final(feature)
        return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        if self.finetune:
            x = self.backbone(x)
            x = self.pooling(x).view(batch_size, -1)
        else:
            self.backbone.eval()
            with torch.no_grad():
                x = self.backbone(x)
                x = self.pooling(x).view(batch_size, -1)
        
        return x

        # if self.use_fc:
        #     x = self.dropout(x)
        #     x = self.fc(x)
        #     x = self.bn(x)
        # return x
        