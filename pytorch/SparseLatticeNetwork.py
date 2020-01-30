import os.path
import sys

#ROOT_DIR = os.path.abspath(os.path.pardir)
#sys.path.append(ROOT_DIR)
import torch
if(torch.cuda.is_available()): import torch.cuda as torch
else: import torch as torch
import torch.nn as nn
from BilateralConvolutionalLayer import BCL

class SplatNet(nn.Module):
    def __init__(self, num_classes):
        super(SplatNet, self).__init__()
        #print("SplatNet Init")
        self.conv1_initial = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(1,1,1), stride=1, padding=0, bias=True)
        self.conv1_f1 = nn.Conv3d(in_channels=352, out_channels=32, kernel_size=(1,1,1), stride=1, padding=0, bias=True)
        self.conv1_f2 = nn.Conv3d(in_channels=32, out_channels=num_classes, kernel_size=(1,1,1), stride=1, padding=0, bias=True)
        
        self.fc1 = nn.Linear(240000, 4000, bias=True)
        self.fc2 = nn.Linear(4000, 2500, bias=True)
        self.leaky = nn.LeakyReLU(0.1)
        
        self.bcl1 = BCL(N=64, C=32, Cp=1)
        self.bcl2 = BCL(N=32, C=64, Cp=32)
        self.bcl3 = BCL(N=32, C=128, Cp=64)
        self.bcl4 = BCL(N=16, C=128, Cp=128)      
    
    def forward(self, data):
        self.data = data
        self.num_points = data.shape[0]
        num_channels = data.shape[1]
        self.prev_data = data.reshape(1, num_channels, self.num_points, 1, 1)
             
        data_conv1 = self.conv1_initial(self.prev_data) 
        out1 = self.bcl1(data, data_conv1)        
        out2 = self.bcl2(data, out1)
        out3 = self.bcl3(data, out2)
        out4 = self.bcl4(data, out3)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.conv1_f1(out)
        out = self.conv1_f2(out)
        
        return out
