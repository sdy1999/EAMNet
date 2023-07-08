
from re import X
from tkinter.ttk import Scale
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from lib.ConvNeXt import *
from lib.Res2Net_v1b import *
from lib.resnet import *
from lib.Rcab import *
from lib.vgg import vgg16



eps = 1e-12

#Channel Reduce
class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel,RFB = False):
        super(Reduction, self).__init__()
        #self.dyConv = Dynamic_conv2d(in_channel,out_channel,3,padding = 1)
        if(RFB):
            self.reduce = nn.Sequential(
                RFB_modified(in_channel,out_channel),
            )
        else:
            self.reduce = nn.Sequential(
                BasicConv2d(in_channel, out_channel, 1),
            )
    def forward(self, x):
        return self.reduce(x)


#
class SEA(nn.Module):
    def __init__(self, channel = 64):
        super(SEA, self).__init__() 
    
        self.upconv2 = conv_upsample()

        self.conv1 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        self.conv2 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        self.conv3 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        self.conv4 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        #self.Bconv1 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        #self.ms = MS_CAM()  
        self.fuse1  = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))
        self.fuse2  = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1))

        self.final_fuse = nn.Sequential(
            ConvBR(channel*2, channel*2,kernel_size=3, stride=1,padding=1),
            BasicConv2d(channel*2, channel,kernel_size=3, stride=1,padding=1),
        )
        
    def forward(self, sen_f, edge_f, edge_previous):   # x guide y
        s1 = F.upsample(sen_f, size=edge_f.size()[2:], mode='bilinear', align_corners=True)
        s1 = self.conv1(s1)  #upsample
        s2 = self.conv2(sen_f) 
        e1 = self.conv3(edge_f)
        e2 = self.conv4(edge_previous)

        e1 = self.fuse1(e1 * s1) + e1
        e2 = self.fuse2(e2 * s2) + e2

        e2  = self.upconv2(e2,e1)  #upsample 

        out  = self.final_fuse(torch.cat((e1,e2),1))

        return out


class Guide_flow(nn.Module):
    def __init__(self, x_channel, y_channel):
        super(Guide_flow, self).__init__()
        self.guidemap = nn.Conv2d(x_channel, 1, 1)

        self.gateconv = GatedSpatailConv2d(y_channel)

    def forward(self, x, y):   # x guide y

        guide = self.guidemap(x)
        guide_flow = F.interpolate(guide, size = y.size()[2:], mode='bilinear') 
        y = self.gateconv(y,guide_flow)

        return y

class GatedSpatailConv2d(nn.Module):
    def __init__(self,channels = 32,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias_attr=False):
        super(GatedSpatailConv2d, self).__init__()
        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(channels+1),
            nn.Conv2d(channels +1, channels +1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels +1, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.conv = nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0,dilation=dilation,groups=groups)

    def forward(self, input_features, gating_features):
        cat = torch.cat((input_features, gating_features),1)
        attention = self._gate_conv(cat)
        x = input_features * (attention + 1)
        x = self.conv(x)
        return x


class IntegralAttention (nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(IntegralAttention, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.conv_cat = nn.Sequential(
            BasicConv2d(in_channel*3, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),

        )
        self.conv_res = BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1)

        self.eps = 1e-5   
        self.IterAtt = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 8, kernel_size=1),
            nn.LayerNorm([out_channel // 8, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // 8, out_channel, kernel_size=1)
        )
        self.ConvOut = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1), 
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        fuse = self.relu(x_cat + self.conv_res(x))

        # can change to the MS-CAM or SE Attention, refer to lib.RCAB.
        context = (fuse.pow(2).sum((2,3), keepdim=True) + self.eps).pow(0.5) # [B, C, 1, 1]
        channel_add_term = self.IterAtt(context)
        out = channel_add_term * fuse + fuse

        out = self.ConvOut(out)

        return out

# integrity aggregate module
class EIA(nn.Module):
    def __init__(self, s_channel = 64, h_channel= 64 ,e_channel= 64 ):
        super(EIA, self).__init__()
        #self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv_1 =  ConvBR(h_channel, h_channel,kernel_size=3, stride=1,padding=1)
        self.conv_2 =  ConvBR(s_channel, h_channel,kernel_size=3, stride=1,padding=1)
        self.conv_3 =  ConvBR(e_channel, e_channel,kernel_size=3, stride=1,padding=1)
        self.conv_d1 = BasicConv2d(h_channel, h_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = BasicConv2d(h_channel, h_channel, kernel_size=3, stride=1, padding=1)

        self.attention = IntegralAttention()
        self.convfuse1 = nn.Sequential(
            BasicConv2d(s_channel*3, s_channel,kernel_size=3, stride=1,padding=1), 
            BasicConv2d(s_channel, s_channel,kernel_size=3, stride=1,padding=1),
        )



    def forward(self, left, down,edge):
        left_1 = self.conv_1(left)
        down_1 = self.conv_2(down)
        #edge_1 = self.conv_3(edge)

        down_2 = self.conv_d1(down_1)
        left_2 = self.conv_l(left_1)

    #z1 conv(down) * left
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear')
        z1 = F.relu(left_1 * down_2, inplace=True)

    #z2 conv(left) * down
        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')
        z2 = F.relu(down_1 * left_2, inplace=True)

        fuse = self.convfuse1(torch.cat((z1, z2, edge), 1))
    #fuse and Integrity enhence

        out = self.attention(fuse)

        return out

class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()
    #  ---- VGG16 Backbone ----
       # self.backbone = eval(vgg16)(pretrained = True)
       # enc_channels=[64, 128, 256, 512, 512]
    #
    #  ---- ConvNext Backbone ----
        #self.backbone = convnext_tiny(pretrained=True)
        #enc_channels=[96, 192, 384,768]
    #
    #   ---- Res2Net50 Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        enc_channels=[256, 512, 1024,2048]
    #
    #   ---- ResNet50 Backbone ----
        #self.resnet = resnet50(pretrained=True)
        #enc_channels=[256, 512, 1024,2048]
    #

        self.reduce_s1 = Reduction(enc_channels[0],channel,RFB = False)
        self.reduce_s2 = Reduction(enc_channels[1],channel,RFB = False)
        self.reduce_s3 = Reduction(enc_channels[2],channel,RFB = False)
        self.reduce_s4 = Reduction(enc_channels[3],channel,RFB = False)

        self.reduce_e1 = Reduction(enc_channels[0],channel,RFB = False)
        self.reduce_e2 = Reduction(enc_channels[1],channel,RFB = False)
        self.reduce_e3 = Reduction(enc_channels[2],channel,RFB = False)
        self.reduce_e4 = Reduction(enc_channels[3],channel,RFB = False)
        
        self.Bconv1 = ConvBR(channel, channel,kernel_size=3, stride=1,padding=1)
        self.Bconv2 = ConvBR(channel, channel,kernel_size=3, stride=1,padding=1)
     
        self.iam1 = EIA(channel,channel,channel)
        self.iam2 = EIA(channel,channel,channel)
        self.iam3 = EIA(channel,channel,channel)

        self.sie1 = SEA()
        self.sie2 = SEA()
        self.sie3 = SEA()

        self.rcab1 = RCAB(64)
        self.rcab2 = RCAB(64)
        self.rcab3 = RCAB(64)
        self.rcab4 = RCAB(64)
        self.rcab5 = RCAB(64)
        self.rcab6 = RCAB(64)


        self.skip1 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1),BasicConv2d(channel, channel,kernel_size=3, stride=1,padding=1))
        self.skip2 = nn.Sequential(ConvBR(channel, channel,kernel_size=3, stride=1,padding=1),BasicConv2d(channel, channel,kernel_size=3, stride=1,padding=1))

        self.guideflow_sh1 = Guide_flow(channel,channel)
        self.guideflow_sh2 = Guide_flow(channel,channel)
        self.guideflow_sh3 = Guide_flow(channel,channel)

        self.guideflow_eh1 = Guide_flow(channel,channel)
        self.guideflow_eh2 = Guide_flow(channel,channel)
        self.guideflow_eh3 = Guide_flow(channel,channel)

        self.pre_out1  = nn.Conv2d(channel, 1, 1)
        self.pre_out2  = nn.Conv2d(channel, 1, 1)
        self.pre_out3  = nn.Conv2d(channel, 1, 1)
        self.pre_out4  = nn.Conv2d(channel, 1, 1)
        self.pre_out5  = nn.Conv2d(channel, 1, 1)
        self.pre_out6  = nn.Conv2d(channel, 1, 1)
        


    def forward(self, x):
    # Feature Extraction
        shape = x.size()[2:]

    #   ---- Res2Net Backbone ---- 
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88

        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
    #
    #  ---- ConvNext Backbone ----
        #datalist = self.backbone(x)
        #x1, x2, x3, x4 = datalist[0], datalist[1], datalist[2], datalist[3]    # bs, 96,48,24,12
    

        
    # Channel Reduce
        x1_t = self.reduce_s1(x1)
        x2_t = self.reduce_s2(x2)
        x3_t = self.reduce_s3(x3)
        x4_t = self.reduce_s4(x4)

        x1_e = self.reduce_e1(x1)
        x2_e = self.reduce_e2(x2)
        x3_e = self.reduce_e3(x3)
        x4_e = self.reduce_e4(x4)
        
    
    #stage 1


        hr_s = self.guideflow_sh1(x4_t,x1_t)
        hr_s = self.rcab1(hr_s)

        hr_e = self.guideflow_eh1(x4_e,x1_e)
        hr_e = self.rcab4(hr_e)

        
        edge_1 = self.sie1(x4_t,x3_e,x4_e)  #24*24
        fuse_1 = self.iam1(x3_t,x4_t,edge_1)  #B*C*24*24

    #stage 2

        hr_s = self.guideflow_sh2(fuse_1,hr_s)
        hr_s = self.rcab2(hr_s)
 
        hr_e = self.guideflow_eh2(edge_1,hr_e)
        hr_e = self.rcab5(hr_e)
  


        edge_2 = self.sie2(fuse_1,x2_e,edge_1) #48*48
        fuse_2 = self.iam2(x2_t,fuse_1,edge_2) #B*C*48*48
        

    #stage 3
        #high_3 = self.guideflow_sh3(fuse_2,high_2)
        #high_3 = self.rcab3(high_3)

        #long skip connection
        hr_s = self.guideflow_sh3(fuse_2,hr_s)
        hr_s = self.rcab3(hr_s)
        #hr_s = self.Bconv1(hr_s) 
        #hr_s = self.skip1(self.Bconv1(hr_s) + x1_t)

        hr_e = self.guideflow_eh3(edge_2,hr_e)
        hr_e = self.rcab6(hr_e)
        #hr_e = self.Bconv2(hr_e)
        #hr_e = self.skip2(self.Bconv2(hr_e) + x1_e)


        edge_3 = self.sie3(fuse_2,hr_e,edge_2)   # 96*96
        fuse_3 = self.iam3(hr_s,fuse_2,edge_3)
        

        preds1   = F.interpolate(self.pre_out1(fuse_1), size = shape, mode='bilinear') 
        preds2   = F.interpolate(self.pre_out2(fuse_2), size = shape, mode='bilinear') 
        pred_f   = F.interpolate(self.pre_out3(fuse_3), size = shape, mode='bilinear')  #final pred


        prede1   = F.interpolate(self.pre_out4(edge_1), size = shape, mode='bilinear') 
        prede2   = F.interpolate(self.pre_out5(edge_2), size = shape, mode='bilinear') 
        prede3   = F.interpolate(self.pre_out6(edge_3), size = shape, mode='bilinear')
    
        return preds1, pred_f, preds2, prede1, prede2, prede3



if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 384, 384)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
