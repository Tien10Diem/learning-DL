import torch
import torch.nn as nn

class block_ResNet(nn.Module):
    def __init__(self, in_channels ,out_channels, stride = 1):
        super().__init__()
        internal_channels = out_channels // 4
        self.main = nn.Sequential(
            nn.Conv2d( in_channels= in_channels, out_channels= internal_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels= internal_channels,out_channels= internal_channels, kernel_size=3, stride= stride, padding=1),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels= internal_channels,out_channels= out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)     
        )   
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels= in_channels,out_channels= out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        self.final_relu = nn.ReLU()
    def forward(self, x):
        out = self.main(x) + self.shortcut(x)
        out = self.final_relu(out)
        return out
    
class MYModel(nn.Module):
    def __init__(self, num_classes=10, num_blocks=[3, 4, 6, 3]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.stage1 = self._make_stage(64, 256, stride=1, num_blocks=num_blocks[0])
        self.stage2 = self._make_stage(256, 512, stride=2, num_blocks= num_blocks[1])
        self.stage3 = self._make_stage(512, 1024, stride=2, num_blocks= num_blocks[2])
        self.stage4 = self._make_stage(1024, 2048, stride=2, num_blocks= num_blocks[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )
        
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        
        self.fc4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        self.fc5 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(),
        )
        
    
    
    
    def _make_stage(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        layers.append(block_ResNet(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block_ResNet(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)

        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    

if __name__ == "__main__":
    model = MYModel()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    print(model)