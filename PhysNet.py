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
from torch.nn.modules.utils import _triple
import torch.nn.functional as F
import torch
import pdb


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

        # self.attention = SelfAttention(64)
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
        # x = self.attention(x)
        x = self.ConvBlock8(F.dropout(x, p=0.2))  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(F.dropout(x, p=0.2))  # x [64, T/4, 8, 8]

        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]
        # h = x.register_hook(self.activations_hook)

        # x = nn.ELU(inplace=True)(x)

        x = self.poolspa(x)  # x [64, T, 1, 1]    -->  groundtruth left and right - 7
        x = self.ConvBlock10(F.dropout(x, p=0.5))  # x [1, T, 1,1]
        print(x.size(), length)
        rPPG = x.view(-1, length)
        print(rPPG.size())
        return rPPG, x_visual, x, x_visual1616

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


class SelfAttention(nn.Module):
    def __init__(self, c, reduction_ratio=16):
        super(SelfAttention, self).__init__()
        self.decoded = nn.Conv3d(c,  math.ceil(c / reduction_ratio), kernel_size=1)
        self.encoded = nn.Conv3d(math.ceil(c / reduction_ratio), c, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        N = x.size()[0]
        C = x.size()[1]

        decoded = self.decoded(x)
        encoded = self.encoded(self.relu(torch.layer_norm(decoded, decoded.size()[1:])))
        encoded = nn.functional.softmax(encoded)
        cnn = x * encoded
        return cnn


########################################
'''        How to use it
    #1. Inference the model
    rPPG, x_visual, x_visual3232, x_visual1616 = model(inputs)

    #2. Normalized the Predicted rPPG signal and GroundTruth BVP signal
    rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize
    BVP_label = (BVP_label-torch.mean(BVP_label)) /torch.std(BVP_label)	 	# normalize

    #3. Calculate the loss
    loss_ecg = Neg_Pearson(rPPG, BVP_label)
'''
########################################


class NegPearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(NegPearson, self).__init__()
        return

    def forward(self, preds, labels):  # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

            # if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            # else:
            #    loss += 1 - torch.abs(pearson)

            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


class NegPeaLoss(nn.Module):
    def __init__(self):
        super(NegPeaLoss, self).__init__()

    def forward(self, x, y):
        if len(x.size()) == 1:
            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(y, 0)
        T = x.shape[1]
        p_coeff = torch.sub(T * torch.sum(torch.mul(x, y), 1), torch.mul(torch.sum(x, 1), torch.sum(y, 1)))
        norm = torch.sqrt((T * torch.sum(x ** 2, 1) - torch.sum(x, 1) ** 2) * (T * torch.sum(y ** 2, 1) - torch.sum(y, 1) ** 2))
        p_coeff = torch.div(p_coeff, norm)
        losses = torch.tensor(1.) - p_coeff
        totloss = torch.mean(losses)
        return totloss
