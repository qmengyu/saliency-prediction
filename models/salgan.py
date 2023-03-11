import torch
import numpy as np
import os
from torch import nn
from torch.autograd import Variable

conv_layer = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D': [512, 512, 512, 'U', 512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64]
}


class Generator(nn.Module):
    def __init__(self, reloder=True):
        super(Generator, self).__init__()
        self.encoder = self.make_conv_layers(conv_layer['E'])
        self.decoder = self.make_deconv_layers(conv_layer['D'])
        self.mymodules = nn.ModuleList([
            self.deconv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        ])

        self.net_params_path = 'weights/gen_modelWeights0090/'
        self.net_params_pathDir = os.listdir(self.net_params_path)
        self.net_params_pathDir.sort()
        if reloder:
            self.load_init_weight()

    def conv2d(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def deconv2d(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def relu(self, inplace=True):  # Change to True?
        return nn.ReLU(inplace)

    def maxpool2d(self, ):
        return nn.MaxPool2d(2)

    def make_conv_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [self.maxpool2d()]
            else:
                conv = self.conv2d(in_channels, v)
                layers += [conv, self.relu(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def make_deconv_layers(self, cfg):
        layers = []
        in_channels = 512
        for v in cfg:
            if v == 'U':
                layers += [nn.Upsample(scale_factor=2)]
            else:
                deconv = self.deconv2d(in_channels, v)
                layers += [deconv, self.relu(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def load_init_weight(self):
        params = self.state_dict()
        n1 = 0
        pretrained_dict = {}
        for k, v in params.items():
            single_file_name = self.net_params_pathDir[n1]
            single_file_path = os.path.join(self.net_params_path, single_file_name)
            pa = np.load(single_file_path)
            pa = torch.from_numpy(pa)
            pretrained_dict[k] = pa
            n1 += 1
        params.update(pretrained_dict)
        self.load_state_dict(params)
        print("SalGAN pretrained model reloader")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.mymodules[0](x)
        x = self.mymodules[1](x)
        return x


# x = Variable(torch.rand([17, 3, 192, 256]))
# print('Input :', x.size())
# net = Generator()
# out = net(x)
# print('Output: ', out.size())
