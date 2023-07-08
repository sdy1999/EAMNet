import torch
import torch.nn as nn
class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False, residual=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        self.residual = residual
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x1 = self.conv(x)
        if self.bn is not None:
            x1 = self.bn(x1)
        if self.residual and x1.shape[1] == x.shape[1]:
            x1 = x + x1
        if self.act is not None:
            x1 = self.act(x1)

        return x1

class VGG16BN(nn.Module):
    def __init__(self):
        super(VGG16BN, self).__init__()
        self.conv1_1 = ConvBNReLU(3, 64, frozen=True)
        self.conv1_2 = ConvBNReLU(64, 64, frozen=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ConvBNReLU(64, 128, frozen=True)
        self.conv2_2 = ConvBNReLU(128, 128, frozen=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = ConvBNReLU(128, 256, frozen=True)
        self.conv3_2 = ConvBNReLU(256, 256, frozen=True)
        self.conv3_3 = ConvBNReLU(256, 256, frozen=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = ConvBNReLU(256, 512, frozen=True)
        self.conv4_2 = ConvBNReLU(512, 512, frozen=True)
        self.conv4_3 = ConvBNReLU(512, 512, frozen=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = ConvBNReLU(512, 512, frozen=True)
        self.conv5_2 = ConvBNReLU(512, 512, frozen=True)
        self.conv5_3 = ConvBNReLU(512, 512, frozen=True)

    def forward(self, input):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        pool3 = self.pool3(conv3_3)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        pool4 = self.pool4(conv4_3)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)

        return conv1_2, conv2_2, conv3_3, conv4_3, conv5_3


def vgg16(pretrained=True):
    model = VGG16BN()
    if pretrained:
        print("loading pretrained/5stages_vgg16_bn-6c64b313.pth")
        model.load_state_dict(torch.load("./models/5stages_vgg16_bn-6c64b313.pth"), strict=False)
    return model