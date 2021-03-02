'''
Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks'
By Zitong Yu, 2019/05/05
If you use the code, please cite:
@inproceedings{yu2019remote,
    title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks},
    author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
    booktitle= {British Machine Vision Conference (BMVC)},
    year = {2019}
}
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
'''

import math
import torch.nn as nn
import torch


class PhysNet(nn.Module):
    """
    PhysNet with 3D convolution model
    """
    def __init__(self, frames=128):
        """
        Initialise PhysNet model
        :param frames: length of sequence to process
        """
        super(PhysNet, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.gcb = GCBlock(64)
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))  # selects one from every frame of input

    def forward(self, x):  # x [3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]

        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x = self.ConvBlock3(x)  # x [32, T, 64,64]

        x = self.MaxpoolSpaTem(x)  # x [32, T/2, 32,32]    Temporal halve

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x = self.ConvBlock5(x)  # x [64, T/2, 32,32]

        x = self.MaxpoolSpaTem(x)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]

        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]
        x = self.gcb(x)
        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]

        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]
        # h = x.register_hook(self.activations_hook)

        # x = nn.ELU(inplace=True)(x)

        x = self.poolspa(x)  # x [64, T, 1, 1]    -->  groundtruth left and right - 7
        x = self.ConvBlock10(x)  # x [1, T, 1,1]
        r_ppg = x.view(-1, length)
        return r_ppg, x_visual, x, x_visual1616

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x = self.ConvBlock3(x)  # x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x)  # x [32, T/2, 32,32]    Temporal halve

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        return x


class GCBlock(nn.Module):
    """
    Global Context Block adapted to 3D convolution
    """
    def __init__(self,  C, reduction_ratio=16):
        """
        Global Context layer
        :param C: number of input channels
        :param reduction_ratio: reduction ratio
        """
        super(GCBlock, self).__init__()
        self.attention = nn.Conv3d(C, out_channels=1, kernel_size=1)
        self.c12 = nn.Conv3d(C, math.ceil(C / reduction_ratio), kernel_size=1)
        self.c15 = nn.Conv3d(math.ceil(C / reduction_ratio), C, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, block_input):
        print(block_input.size())
        N = block_input.size()[0]
        C = block_input.size()[1]
        D = block_input.size()[2]

        attention = self.attention(block_input)
        print(attention.size())
        block_input = nn.functional.softmax(block_input)

        block_input_flattened = torch.reshape(block_input, [N, C, D, -1])
        attention = torch.squeeze(attention, dim=3)
        attention_flattened = torch.reshape(attention, [N, D, -1])

        c11 = torch.einsum('bcdf,bdf->bcd', block_input_flattened,
                           attention_flattened)
        c11 = torch.reshape(c11, (N, C, D, 1, 1))
        c12 = self.c12(c11)

        c15 = self.c15(self.relu(torch.layer_norm(c12, c12.size()[1:])))
        cnn = torch.add(block_input, c15)
        return cnn
