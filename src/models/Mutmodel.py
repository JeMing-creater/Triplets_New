import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import os

class TripletModel(nn.Module):
    def __init__(self, model_name, class_num=131, pretrained=True):
        super().__init__()

        """
        Models class to return swin transformer models
        """

        # Load the backbone
        self.model = timm.create_model(model_name, pretrained=pretrained)


        # Get the number features in final embedding
        n_features = self.model.head.in_features

        # Update the classification layer with our custom target size
        self.model.head = nn.Linear(n_features, class_num)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    device = 'cuda:0'
    
    image = torch.randn(2, 3, 224, 224).to(device)
    
    model = TripletModel(model_name='swin_base_patch4_window7_224').to(device)
    
    output = model(image)
    
    triple = output[:, :100]
    tool = output[:, 100:106]
    verb = output[:, 106:116]
    target = output[:, 116:]
    print(output.shape)