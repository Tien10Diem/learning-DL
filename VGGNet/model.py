import torch
import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self, num_classes=10, numbers_blocks = [2, 2, 2, 3, 3]):
        super().__init__()
        self.conv1 = self._make_stage(in_channels= 3, out_channels= 64, numbers_blocks= numbers_blocks[0])
        self.conv2 = self._make_stage(in_channels= 64, out_channels= 128, numbers_blocks= numbers_blocks[1])
        self.conv3 = self._make_stage(in_channels= 128, out_channels= 256, numbers_blocks= numbers_blocks[2])
        self.conv4 = self._make_stage(in_channels= 256, out_channels= 512, numbers_blocks= numbers_blocks[3])
        self.conv5 = self._make_stage(in_channels= 512, out_channels= 512, numbers_blocks= numbers_blocks[4])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(512,256) ,
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256,num_classes) 
        )
        
        
    def _make_stage(self, in_channels,out_channels, numbers_blocks):
        
        layers = []
        layers.append(block_VGG(in_channels= in_channels, out_channels= out_channels))
        
        for i in range(numbers_blocks-1):
            layers.append(block_VGG(in_channels= out_channels, out_channels= out_channels))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
        
class block_VGG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = self._make_block(in_channels= in_channels, out_channels= out_channels)
    
    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, stride = 1, padding= 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )    
    
    def forward(self, x):
        x = self.block(x)
        return x
    
if __name__ == "__main__":
    model = VGGNet()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    print(model)