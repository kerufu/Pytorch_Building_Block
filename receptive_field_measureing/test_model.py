import torch
import torch.nn as nn

from measureing_function import meansure

class testmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, (3, 5), stride=2, dilation=2),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 6, (5, 3), dilation=3)
        )

    def forward(self, x):
        return self.layers(x)
    
model = testmodel()

def init_weights(m):
    try:
        torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        m.bias.data.fill_(0.01)
    except:
        pass

model.apply(init_weights)
meansure(model)