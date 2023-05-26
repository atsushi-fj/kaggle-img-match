import torch
import torch.nn as nn
import timm


class Effnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.effnet = timm.create_model("tf_efficientnet_b7", pretrained=True)
        for param in self.effnet.parameters():
            param.requires_grad = False
        self.rotation_out = nn.Linear(1000, 9)
        self.translation_out = torch.nn.Linear(1000, 3)
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x):
        x = self.effnet(x)
        rotation_out = self.rotation_out(x)
        rotation_out = self.tanh(rotation_out)
        
        translation_out = self.translation_out(x)
        
        return rotation_out, translation_out
        