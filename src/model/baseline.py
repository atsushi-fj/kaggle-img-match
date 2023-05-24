import torch
import torch.nn as nn


class BaseCNN(torch.nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.layer_one = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer_two = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer_three = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer_four = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer_five = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2), 
            nn.AvgPool2d(kernel_size=7))
        
        self.layer_six = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.rotation_out = nn.Linear(512, 9)
        self.tanh = torch.nn.Tanh()
        
        self.translation_out = torch.nn.Linear(512, 3)
        
    def forward(self, x):
        x = self.layer_one(x)
        x = self.layer_two(x)
        x = self.layer_three(x)
        x = self.layer_four(x)
        x = self.layer_five(x)
        
        x = x.view(-1, 512)
        x = self.layer_six(x)
        
        rotation_out = self.rotation_out(x)
        rotation_out = self.tanh(rotation_out)
        
        translation_out = self.translation_out(x)
        
        return rotation_out, translation_out
    