import torch
import torch.nn as nn



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.flatten = nn.Flatten() # flatten function
        
        #Layers
        '''
        28x28x1 -> 26x26x8 -> 13x13x8 -> 10
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.fcLayer = nn.Linear(7 * 7 * 32, 10)
        
    def forward(self, img):
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.flatten(img)
        img = self.fcLayer(img)
        return img
        