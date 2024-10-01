import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LinearClassifierToken(nn.Module):
    def __init__(self, in_channels, num_chanel=2, tokenW=32, tokenH=32):
        super(LinearClassifierToken, self).__init__()
        self.in_channels=in_channels
        self.W=tokenW
        self.H=tokenH
        self.nc=num_chanel
        self.conv=torch.nn.Conv2d(in_channels,num_chanel,(1,1))
    def forward(self,x):
        return self.conv(x.reshape(-1,self.H,self.W,self.in_channels).permute(0,3,1,2))
        
class DinoV2(nn.Module):
    def __init__(self, num_class=16) -> None:
        super().__init__()
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.dinov2.parameters():
            param.requires_grad = False
        n=512
        self.classlayer_224 = LinearClassifierToken(in_channels=384,num_chanel=n,tokenW=16,tokenH=16)
        self.selu = nn.SELU()
        self.to_224 = nn.Sequential(
            nn.Conv2d(n,n,kernel_size=5,stride=1,padding=1,bias=False),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n,n//2,kernel_size=3,stride=1,padding=1,bias=False),
		    nn.BatchNorm2d(n//2),
			nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n//2,n//4,kernel_size=3,stride=1,padding=1,bias=False),
		    nn.BatchNorm2d(n//4),
			nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n//4,n//8,kernel_size=3,stride=1,padding=1,bias=False),
		    nn.BatchNorm2d(n//8),
			nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n//8,n//16,kernel_size=3,stride=1,padding=1,bias=False),
			nn.ReLU(inplace=True)
        )
        self.conv2class = nn.Conv2d(n//16,num_class,kernel_size=3,stride=1,padding=1,bias=True)


    def forward(self, x):
        with torch.no_grad():
            device = next(self.classlayer_224.parameters()).device
            features = self.dinov2.forward_features(x.to(device))['x_norm_patchtokens']
        x = self.selu(self.classlayer_224(features))
        x = self.to_224(x)
        x = self.conv2class(x)
        return x
    
if __name__ == '__main__':
    model = DinoV2(16)
    model = model.to("cuda")

    example_input = torch.rand(1, 3, 224, 224)

    # Run the model
    output = model(example_input)

    # Print the output shape
    print(output.shape)