"""
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
"""
import torch.nn as nn


class PhysNet(nn.Module):
    """
    PhysNet model with spatio-temporal convolution.
    """
    def __init__(self):
        super(PhysNet, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        #######################
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(32, 32, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        ########################
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(32, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        ########################
        self.ConvBlock10 = nn.Sequential(
            nn.Conv3d(64, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock11 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock12 = nn.Sequential(
            nn.Conv3d(64, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock13 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        ########################
        self.ConvBlock14 = nn.Sequential(  # padding?? padding=[0, 2, 2]
            nn.Conv3d(64, 64, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock15 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock16 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.AvgpoolSpa1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.AvgpoolSpa2 = nn.AvgPool3d((1, 7, 7), stride=(1, 2, 2))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):  # x [3, T, 120,120]
        x_visual = x
        [batch, channel, length, width, height] = x.shape
        x = self.ConvBlock1(x)  # x [3, T, 120,120]

        x = self.MaxpoolSpa(x)  # x [16, T, 60,60]

        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)

        x = self.MaxpoolSpa(x)  # x [32, T-4, 30, 30]

        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)  # ------- x [64, T-6, 15, 15]
        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)

        x = self.MaxpoolSpa(x)  # x [64, T-8, 15, 15]

        x = self.ConvBlock10(x)
        x = self.ConvBlock11(x)
        x = self.ConvBlock12(x)  # ------ x [64, T-8, 7, 7]
        x = self.ConvBlock13(x)

        x = self.AvgpoolSpa1(x)  # x [64, T-12, 7, 7]

        x = self.ConvBlock14(x)  # x [64, T-12, 7, 7]
        # h = x.register_hook(self.activations_hook)

        x = self.ConvBlock15(x)  # x [64, T-14, 7, 7] ---- T-10
        x = self.AvgpoolSpa2(x)  # x [64, T-14, 1, 1]
        x = self.ConvBlock16(x)  # x [1, T-14, 1,1]
        rPPG = x.view(-1, length)

        return rPPG, x_visual

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.ConvBlock1(x)  # x [3, T, 120,120]
        x = self.AvgpoolSpa1(x)  # x [16, T, 60,60]

        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.AvgpoolSpa1(x)  # x [32, T-4, 30, 30]

        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)  # ------- x [64, T-6, 15, 15]
        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.AvgpoolSpa1(x)  # x [64, T-8, 15, 15]

        x = self.ConvBlock10(x)
        x = self.ConvBlock11(x)
        x = self.ConvBlock12(x)  # ------ x [64, T-8, 7, 7]
        x = self.ConvBlock13(x)
        x = self.AvgpoolSpa1(x)  # x [64, T-12, 7, 7]

        x = self.ConvBlock14(x)  # x [64, T-12, 7, 7]
        x = self.ConvBlock15(x)  # x [64, T-14, 7, 7] ---- T-10

        return x



