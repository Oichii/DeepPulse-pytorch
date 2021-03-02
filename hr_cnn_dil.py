import torch.nn as nn
import torch.nn.functional as F


class HrCNN(nn.Module):
    def __init__(self, rgb):
        super(HrCNN, self).__init__()
        self.rgb = rgb

        self.ada_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=(192, 128))

        conv_init_mean = 0
        conv_init_std = .1
        xavier_normal_gain = 1

        self.bn_input = nn.BatchNorm2d(self.rgb)
        nn.init.normal_(self.bn_input.weight, conv_init_mean, conv_init_std)

        input_count = self.rgb

        output_count = 64
        self.conv_00 = nn.Conv2d(input_count, output_count, kernel_size=(15, 10), stride=1, padding=0, dilation=1)
        nn.init.xavier_normal_(self.conv_00.weight, gain=xavier_normal_gain)
        self.max_pool2d_00 = nn.MaxPool2d(kernel_size=(15, 10), stride=(2, 2), )
        self.bn_00 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_00.weight, conv_init_mean, conv_init_std)

        input_count = 64
        self.conv_01 = nn.Conv2d(input_count, output_count, kernel_size=(8, 6), stride=1, padding=(0, 1), dilation=2)
        nn.init.xavier_normal_(self.conv_01.weight, gain=xavier_normal_gain)
        self.max_pool2d_01 = nn.MaxPool2d(kernel_size=(15, 10), stride=(1, 1))
        self.bn_01 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_01.weight, conv_init_mean, conv_init_std)

        output_count = 128#64
        self.conv_10 = nn.Conv2d(input_count, output_count, kernel_size=(8, 6), stride=1, padding=0, dilation=2)
        nn.init.xavier_normal_(self.conv_10.weight, gain=xavier_normal_gain)
        self.max_pool2d_10 = nn.MaxPool2d(kernel_size=(15, 10), stride=(1, 1))
        self.bn_10 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_10.weight, conv_init_mean, conv_init_std)

        input_count = 128#64

        output_count = 128#64
        self.conv_20 = nn.Conv2d(input_count, output_count, kernel_size=(12, 10), stride=1, padding=0, dilation=1)
        nn.init.xavier_normal_(self.conv_20.weight, gain=xavier_normal_gain)
        self.max_pool2d_20 = nn.MaxPool2d(kernel_size=(15, 10), stride=(1, 1))
        self.bn_20 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_20.weight, conv_init_mean, conv_init_std)

        input_count = 128#64
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
        x = self.conv_20(F.dropout2d(x, p=0.2, training=self.training))

        x = nonlin(self.bn_20(self.max_pool2d_20(x)))
        x = self.conv_last(F.dropout(x, p=0.5, training=self.training))

        if sum(x.size()[1:]) > x.dim() - 1:
            print(x.size()[1:], x.dim())
            raise ValueError('Layers size error')

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
