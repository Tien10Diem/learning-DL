import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self._make_block(in_channels=3, out_channels=8)
        self.conv2 = self._make_block(in_channels=8, out_channels=16)
        self.conv3 = self._make_block(in_channels=16, out_channels=32)
        self.conv4 = self._make_block(in_channels=32, out_channels=64)
        self.conv5 = self._make_block(in_channels=64, out_channels=128)
        
        self.fc1 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(6272,512) ,
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(512,1024) ,
            nn.LeakyReLU()
        )
        
        self.fc3 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(1024,10) ,
        )
     
    def _make_block(self, in_channels, out_channels):   
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, stride = 1, padding= 'same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=3, stride = 1, padding= 'same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
            )  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x=x.view(x.shape[0], -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
    

if __name__ == "__main__":
    model = simpleCNN()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out.shape)