import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class HrCNN(nn.Module):
    def __init__(self, rgb=3):
        super(HrCNN, self).__init__()
        self.rgb = rgb

        self.ada_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=(192, 128))

        conv_init_mean = 0
        conv_init_std = .1
        xavier_normal_gain = 1
        input_count = rgb

        self.bn_input = nn.BatchNorm2d(input_count)
        nn.init.normal_(self.bn_input.weight, conv_init_mean, conv_init_std)

        output_count = 64
        self.conv_00 = nn.Conv2d(input_count, output_count, kernel_size=(15, 10), stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_00.weight, gain=xavier_normal_gain)
        self.max_pool2d_00 = nn.MaxPool2d(kernel_size=(15, 10), stride=(2, 2), )
        self.bn_00 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_00.weight, conv_init_mean, conv_init_std)

        input_count = 64

        self.conv_01 = nn.Conv2d(input_count, output_count, kernel_size=(15, 10), stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_01.weight, gain=xavier_normal_gain)
        self.max_pool2d_01 = nn.MaxPool2d(kernel_size=(15, 10), stride=(1, 1))
        self.bn_01 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_01.weight, conv_init_mean, conv_init_std)

        output_count = 128
        self.conv_10 = nn.Conv2d(input_count, output_count, kernel_size=(15, 10), stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_10.weight, gain=xavier_normal_gain)
        self.max_pool2d_10 = nn.MaxPool2d(kernel_size=(15, 10), stride=(1, 1))
        self.bn_10 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_10.weight, conv_init_mean, conv_init_std)

        input_count = 128

        output_count = 128

        self.gcb = GCBlock(output_count)

        self.conv_20 = nn.Conv2d(input_count, output_count, kernel_size=(12, 10), stride=1, padding=0)
        self.max_pool2d_20 = nn.MaxPool2d(kernel_size=(15, 10), stride=(1, 1))
        self.bn_20 = nn.BatchNorm2d(output_count)

        input_count = 128
        self.conv_last = nn.Conv2d(input_count, 1, kernel_size=1, stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_last.weight, gain=xavier_normal_gain)

        self.gradients = None

    def forward(self, x):
        nonlin = F.elu

        x = self.ada_avg_pool2d(x)

        x = self.bn_input(x)
        x = nonlin(self.bn_00(self.max_pool2d_00(self.conv_00(F.dropout2d(x, p=0.0, training=self.training)))))
        x = nonlin(self.bn_01(self.max_pool2d_01(self.conv_01(F.dropout(x, p=0.0, training=self.training)))))
        x = nonlin(self.bn_10(self.max_pool2d_10(self.conv_10(F.dropout(x, p=0.0, training=self.training)))))

        # h = x.register_hook(self.activations_hook)

        x = self.gcb(x)  # <--------- global convolution block
        x_features = self.conv_20(F.dropout2d(x, p=0.2, training=self.training))
        x = nonlin(self.bn_20(self.max_pool2d_20(x_features)))

        x = self.conv_last(F.dropout(x, p=0.5, training=self.training))

        if sum(x.size()[1:]) > x.dim() - 1:
            raise ValueError('Check your network!')

        return x

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        nonlin = F.elu
        x = self.ada_avg_pool2d(x)
        x = self.bn_input(x)
        x = nonlin(self.bn_00(self.max_pool2d_00(self.conv_00(F.dropout2d(x, p=0.0, training=self.training)))))
        x = nonlin(self.bn_01(self.max_pool2d_01(self.conv_01(F.dropout(x, p=0.0, training=self.training)))))
        x = nonlin(self.bn_10(self.max_pool2d_10(self.conv_10(F.dropout(x, p=0.0, training=self.training)))))
        return x


class GCBlock(nn.Module):
    """
    Global Context block
    """
    def __init__(self, c, reduction_ratio=16):
        """
        Initialize global context layer
        :param c: number of input channels
        :param reduction_ratio: reduction ratio, default 16
        """
        super(GCBlock, self).__init__()
        self.attention = nn.Conv2d(c, out_channels=1, kernel_size=1)
        self.c12 = nn.Conv2d(c, math.ceil(c / reduction_ratio), kernel_size=1)
        self.c15 = nn.Conv2d(math.ceil(c / reduction_ratio), c, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, block_input):
        N = block_input.size()[0]
        C = block_input.size()[1]

        attention = self.attention(block_input)

        block_input = nn.functional.softmax(block_input)

        block_input_flattened = torch.reshape(block_input, [N, C, -1])
        attention = torch.squeeze(attention, dim=3)
        attention_flattened = torch.reshape(attention, [N, -1])

        c11 = torch.einsum('bcf,bf->bc', block_input_flattened,
                           attention_flattened)
        c11 = torch.reshape(c11, (N, C, 1, 1))

        c12 = self.c12(c11)

        c15 = self.c15(self.relu(torch.layer_norm(c12, c12.size()[1:])))
        cnn = torch.add(block_input, c15)
        return cnn


class SelfAttention(nn.Module):
    """
    SelfAttention block
    """
    def __init__(self, c, reduction_ratio=16):
        """
        Initialize SelfAttention layer
        :param c: number of channels
        :param reduction_ratio: reduction ratio, default 16
        """
        super(SelfAttention, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.decoded = nn.Conv2d(c,  math.ceil(c / reduction_ratio), kernel_size=1)
        self.encoded = nn.Conv2d(math.ceil(c / reduction_ratio), c, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        N = x.size()[0]
        C = x.size()[1]
        xx = self.pooling(x)
        decoded = self.decoded(xx)
        encoded = self.encoded(self.relu(torch.layer_norm(decoded, decoded.size()[1:])))
        encoded = nn.functional.softmax(encoded)
        cnn = x * encoded
        return cnn


