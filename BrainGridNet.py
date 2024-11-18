"""
BrainGridNet_PSD32 and BrainGridNet_Raw are the experimental models used in our study, while others serve as ablation comparison models.

If you find this code helpful, please consider citing:
BrainGridNet: A two-branch depthwise CNN for decoding EEG-based multi-class motor imagery

For any questions, feel free to reach out to me at wangxingfu21@mails.ucas.ac.cn.
Thank you for your support!
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BrainGridNet_PSD32(nn.Module):
    """
    Designed for processing PSD data of EEG signals.
    Note: Applying `10 * np.log10()` to the PSD data is essential;
    otherwise, the model may fail to capture the discriminative features of high-frequency components.

    Input shape: (batch_size, 9, 9, 32)
    Output shape: (batch_size, num_classes)

    For detailed shape descriptions, please refer to the paper.

    """

    def __init__(self):
        super(BrainGridNet_PSD32, self).__init__()

        # Branch1
        self.conv0 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        self.padding1 = nn.ZeroPad2d((2, 3, 1, 1))
        self.conv1 = nn.Conv2d(9, 72, (3, 6), groups=9)
        self.batchnorm1 = nn.BatchNorm2d(72, False)
        # self.laynorm1 = nn.LayerNorm([72, 9, 32])
        self.pooling1 = nn.AvgPool2d((3, 6))
        self.padding2 = nn.ZeroPad2d((1, 1, 1, 0))
        self.conv2 = nn.Conv2d(72, 72, (2, 3), groups=72)
        self.batchnorm2 = nn.BatchNorm2d(72, False)
        # self.laynorm2 = nn.LayerNorm([72, 3, 5])
        self.pooling2 = nn.AvgPool2d((2, 4))

        # Branch2
        self.conv0_2 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        self.padding1_2 = nn.ZeroPad2d((2, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(9, 72, (3, 6), groups=9)
        self.batchnorm1_2 = nn.BatchNorm2d(72, False)
        # self.laynorm1_2 = nn.LayerNorm([72, 9, 32])
        self.pooling1_2 = nn.AvgPool2d((3, 6))
        self.padding2_2 = nn.ZeroPad2d((1, 1, 1, 0))
        self.conv2_2 = nn.Conv2d(72, 72, (2, 3), groups=72)
        self.batchnorm2_2 = nn.BatchNorm2d(72, False)
        # self.laynorm2_2 = nn.LayerNorm([72, 3, 5])
        self.pooling2_2 = nn.AvgPool2d((2, 4))

        # Dense
        self.fc = nn.Linear(144, 5)

    def forward(self, x):
        x1 = x
        x2 = x.permute(0, 2, 1, 3)
        x1 = F.elu(self.conv0(x1))
        x2 = F.elu(self.conv0_2(x2))
        # Branch1
        tensor_1 = F.dropout(x1, 0.25)
        tensor_1 = self.padding1(tensor_1)
        tensor_1 = F.elu(self.conv1(tensor_1))
        tensor_1 = self.batchnorm1(tensor_1)
        tensor_1 = F.dropout(tensor_1, 0.25)
        tensor_1 = self.pooling1(tensor_1)
        tensor_1 = self.padding2(tensor_1)
        tensor_1 = F.elu((self.conv2(tensor_1)))
        tensor_1 = self.batchnorm2(tensor_1)
        tensor_1 = self.pooling2(tensor_1)
        # Branch2
        tensor_2 = F.dropout(x2, 0.25)
        tensor_2 = self.padding1_2(tensor_2)
        tensor_2 = F.elu(self.conv1_2(tensor_2))
        tensor_2 = self.batchnorm1_2(tensor_2)
        tensor_2 = F.dropout(tensor_2, 0.25)
        tensor_2 = self.pooling1_2(tensor_2)
        tensor_2 = self.padding2_2(tensor_2)
        tensor_2 = F.elu((self.conv2_2(tensor_2)))
        tensor_2 = self.batchnorm2_2(tensor_2)
        tensor_2 = self.pooling2_2(tensor_2)
        # Concat
        out = torch.cat([tensor_1, tensor_2], dim=1)
        x = out.view(x.shape[0], 144)
        x = torch.softmax(self.fc(x), dim=1)
        return x


class BrainGridNet_PSD32_Branch1(nn.Module):

    def __init__(self):
        super(BrainGridNet_PSD32_Branch1, self).__init__()

        # Branch1
        self.conv0 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        self.padding1 = nn.ZeroPad2d((2, 3, 1, 1))
        self.conv1 = nn.Conv2d(9, 72, (3, 6), groups=9)
        self.batchnorm1 = nn.BatchNorm2d(72, False)
        self.pooling1 = nn.AvgPool2d((3, 6))
        self.padding2 = nn.ZeroPad2d((1, 1, 1, 0))
        self.conv2 = nn.Conv2d(72, 72, (2, 3), groups=72)
        self.batchnorm2 = nn.BatchNorm2d(72, False)
        self.pooling2 = nn.AvgPool2d((2, 4))

        # Branch2
        self.conv0_2 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        self.padding1_2 = nn.ZeroPad2d((2, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(9, 72, (3, 6), groups=9)
        self.batchnorm1_2 = nn.BatchNorm2d(72, False)
        self.pooling1_2 = nn.AvgPool2d((3, 6))
        self.padding2_2 = nn.ZeroPad2d((1, 1, 1, 0))
        self.conv2_2 = nn.Conv2d(72, 72, (2, 3), groups=72)
        self.batchnorm2_2 = nn.BatchNorm2d(72, False)
        self.pooling2_2 = nn.AvgPool2d((2, 4))

        # Dense
        self.fc = nn.Linear(72, 5)  # 144->72

    def forward(self, x):
        x1 = x
        # x2 = x.permute(0, 2, 1, 3)
        x1 = F.elu(self.conv0(x1))
        # x2 = F.elu(self.conv0_2(x2))
        # Branch1
        tensor_1 = F.dropout(x1, 0.25)
        tensor_1 = self.padding1(tensor_1)
        tensor_1 = F.elu(self.conv1(tensor_1))
        tensor_1 = self.batchnorm1(tensor_1)
        tensor_1 = F.dropout(tensor_1, 0.25)
        tensor_1 = self.pooling1(tensor_1)
        tensor_1 = self.padding2(tensor_1)
        tensor_1 = F.elu((self.conv2(tensor_1)))
        tensor_1 = self.batchnorm2(tensor_1)
        tensor_1 = self.pooling2(tensor_1)
        # Branch2
        # tensor_2 = F.dropout(x2, 0.25)
        # tensor_2 = self.padding1_2(tensor_2)
        # tensor_2 = F.elu(self.conv1_2(tensor_2))
        # tensor_2 = self.batchnorm1_2(tensor_2)
        # tensor_2 = F.dropout(tensor_2, 0.25)
        # tensor_2 = self.pooling1_2(tensor_2)
        # tensor_2 = self.padding2_2(tensor_2)
        # tensor_2 = F.elu((self.conv2_2(tensor_2)))
        # tensor_2 = self.batchnorm2_2(tensor_2)
        # tensor_2 = self.pooling2_2(tensor_2)
        # Concat
        # out = torch.cat([tensor_1, tensor_2], dim=1)
        x = tensor_1.view(tensor_1.shape[0], 72)
        x = torch.softmax(self.fc(x), dim=1)
        return x


class BrainGridNet_PSD32_Branch2(nn.Module):

    def __init__(self):
        super(BrainGridNet_PSD32_Branch2, self).__init__()

        # Branch1
        self.conv0 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        self.padding1 = nn.ZeroPad2d((2, 3, 1, 1))
        self.conv1 = nn.Conv2d(9, 72, (3, 6), groups=9)
        self.batchnorm1 = nn.BatchNorm2d(72, False)
        self.pooling1 = nn.AvgPool2d((3, 6))
        self.padding2 = nn.ZeroPad2d((1, 1, 1, 0))
        self.conv2 = nn.Conv2d(72, 72, (2, 3), groups=72)
        self.batchnorm2 = nn.BatchNorm2d(72, False)
        self.pooling2 = nn.AvgPool2d((2, 4))

        # Branch2
        self.conv0_2 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        self.padding1_2 = nn.ZeroPad2d((2, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(9, 72, (3, 6), groups=9)
        self.batchnorm1_2 = nn.BatchNorm2d(72, False)
        # self.laynorm1_2 = nn.LayerNorm([72, 9, 32])
        self.pooling1_2 = nn.AvgPool2d((3, 6))
        self.padding2_2 = nn.ZeroPad2d((1, 1, 1, 0))
        self.conv2_2 = nn.Conv2d(72, 72, (2, 3), groups=72)
        self.batchnorm2_2 = nn.BatchNorm2d(72, False)
        self.pooling2_2 = nn.AvgPool2d((2, 4))

        # Dense
        self.fc = nn.Linear(72, 5)  # 144->72

    def forward(self, x):
        # x1 = x
        x2 = x.permute(0, 2, 1, 3)
        # x1 = F.elu(self.conv0(x1))
        x2 = F.elu(self.conv0_2(x2))
        # Branch1
        # tensor_1 = F.dropout(x1, 0.25)
        # tensor_1 = self.padding1(tensor_1)
        # tensor_1 = F.elu(self.conv1(tensor_1))
        # tensor_1 = self.batchnorm1(tensor_1)
        # tensor_1 = F.dropout(tensor_1, 0.25)
        # tensor_1 = self.pooling1(tensor_1)
        # tensor_1 = self.padding2(tensor_1)
        # tensor_1 = F.elu((self.conv2(tensor_1)))
        # tensor_1 = self.batchnorm2(tensor_1)
        # tensor_1 = self.pooling2(tensor_1)
        # Branch2
        tensor_2 = F.dropout(x2, 0.25)
        tensor_2 = self.padding1_2(tensor_2)
        tensor_2 = F.elu(self.conv1_2(tensor_2))
        tensor_2 = self.batchnorm1_2(tensor_2)
        tensor_2 = F.dropout(tensor_2, 0.25)
        tensor_2 = self.pooling1_2(tensor_2)
        tensor_2 = self.padding2_2(tensor_2)
        tensor_2 = F.elu((self.conv2_2(tensor_2)))
        tensor_2 = self.batchnorm2_2(tensor_2)
        tensor_2 = self.pooling2_2(tensor_2)
        # Concat
        x = tensor_2.view(tensor_2.shape[0], 72)  # 144->72, because only one branch is used
        x = torch.softmax(self.fc(x), dim=1)
        return x


class BrainGridNet_PSD32_without_depthwise(nn.Module):

    def __init__(self):
        super(BrainGridNet_PSD32_without_depthwise, self).__init__()

        # Branch1
        self.conv0 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0))  # , groups=9
        self.padding1 = nn.ZeroPad2d((2, 3, 1, 1))
        self.conv1 = nn.Conv2d(9, 72, (3, 6))  # , groups=9
        self.batchnorm1 = nn.BatchNorm2d(72, False)
        self.pooling1 = nn.AvgPool2d((3, 6))
        self.padding2 = nn.ZeroPad2d((1, 1, 1, 0))
        self.conv2 = nn.Conv2d(72, 72, (2, 3))  # , groups=72
        self.batchnorm2 = nn.BatchNorm2d(72, False)
        self.pooling2 = nn.AvgPool2d((2, 4))

        # Branch2
        self.conv0_2 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0))  # , groups=9
        self.padding1_2 = nn.ZeroPad2d((2, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(9, 72, (3, 6))  # , groups=9
        self.batchnorm1_2 = nn.BatchNorm2d(72, False)
        self.pooling1_2 = nn.AvgPool2d((3, 6))
        self.padding2_2 = nn.ZeroPad2d((1, 1, 1, 0))
        self.conv2_2 = nn.Conv2d(72, 72, (2, 3))  # , groups=72
        self.batchnorm2_2 = nn.BatchNorm2d(72, False)
        self.pooling2_2 = nn.AvgPool2d((2, 4))

        # Dense
        self.fc = nn.Linear(144, 5)

    def forward(self, x):
        x1 = x
        x2 = x.permute(0, 2, 1, 3)
        x1 = F.elu(self.conv0(x1))
        x2 = F.elu(self.conv0_2(x2))
        # Branch1
        tensor_1 = F.dropout(x1, 0.25)
        tensor_1 = self.padding1(tensor_1)
        tensor_1 = F.elu(self.conv1(tensor_1))
        tensor_1 = self.batchnorm1(tensor_1)
        tensor_1 = F.dropout(tensor_1, 0.25)
        tensor_1 = self.pooling1(tensor_1)
        tensor_1 = self.padding2(tensor_1)
        tensor_1 = F.elu((self.conv2(tensor_1)))
        tensor_1 = self.batchnorm2(tensor_1)
        tensor_1 = self.pooling2(tensor_1)
        # Branch2
        tensor_2 = F.dropout(x2, 0.25)
        tensor_2 = self.padding1_2(tensor_2)
        tensor_2 = F.elu(self.conv1_2(tensor_2))
        tensor_2 = self.batchnorm1_2(tensor_2)
        tensor_2 = F.dropout(tensor_2, 0.25)
        tensor_2 = self.pooling1_2(tensor_2)
        tensor_2 = self.padding2_2(tensor_2)
        tensor_2 = F.elu((self.conv2_2(tensor_2)))
        tensor_2 = self.batchnorm2_2(tensor_2)
        tensor_2 = self.pooling2_2(tensor_2)
        # Concat
        out = torch.cat([tensor_1, tensor_2], dim=1)
        x = out.view(x.shape[0], 144)
        x = torch.softmax(self.fc(x), dim=1)
        return x


class BrainGridNet_PSD32_share(nn.Module):
    '''
    Share the weights of the two branches.
    '''

    def __init__(self):
        super(BrainGridNet_PSD32_share, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        # Layer 2
        self.padding1 = nn.ZeroPad2d((1, 1, 2, 3))
        self.conv2 = nn.Conv2d(9, 72, (6, 3), groups=9)
        self.batchnorm1 = nn.BatchNorm2d(72, False)
        self.laynorm1 = nn.LayerNorm([72, 32, 9])
        self.pooling2 = nn.AvgPool2d((6, 3))
        # Layer 3
        self.padding2 = nn.ZeroPad2d((0, 1, 1, 1))
        self.conv3 = nn.Conv2d(72, 72, (3, 2), groups=72)
        self.batchnorm2 = nn.BatchNorm2d(72, False)
        self.laynorm2 = nn.LayerNorm([72, 5, 3])
        self.pooling3 = nn.AvgPool2d((4, 2))
        # FC Layer
        self.fc1 = nn.Linear(144, 5)

    def forward(self, x):
        # Layer 1
        # x shape (bs,9,9,64)
        x1 = x
        x2 = x.permute(0, 2, 1, 3)
        x1 = F.elu(self.conv1(x1)).contiguous().view(-1, 9, 32, 9)
        x2 = F.elu(self.conv1(x2)).contiguous().view(-1, 9, 32, 9)
        out = []
        for index, tensor in enumerate([x1, x2]):
            tensor = F.dropout(tensor, 0.25)
            tensor = self.padding1(tensor)
            tensor = F.elu(self.conv2(tensor))
            tensor = self.laynorm1(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling2(tensor)
            tensor = self.padding2(tensor)
            tensor = F.elu((self.conv3(tensor)))
            tensor = self.laynorm2(tensor)
            tensor = self.pooling3(tensor)

            if index == 0:
                out = tensor
            elif index == 1 or 2:
                out = torch.cat([out, tensor], dim=1)
        x = out.contiguous().view(4, 144)
        x = torch.softmax(self.fc1(x), dim=1)
        return x


class BrainGridNet_Raw(nn.Module):

    def __init__(self):
        super(BrainGridNet_Raw, self).__init__()
        self.conv1 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        self.padding1 = nn.ZeroPad2d((7, 8, 1, 2))
        self.conv2 = nn.Conv2d(9, 36, (4, 16), groups=9)
        self.laynorm1 = nn.LayerNorm([36, 9, 655])
        self.pooling1 = nn.MaxPool2d((3, 6))
        self.padding2 = nn.ZeroPad2d((3, 4, 0, 1))
        self.conv3 = nn.Conv2d(36, 36, (2, 8), groups=36)
        self.laynorm2 = nn.LayerNorm([36, 3, 109])
        self.pooling2 = nn.MaxPool2d((2, 4))
        self.fc1 = nn.Linear(1944, 5)

    def forward(self, x):
        x1 = x
        x2 = x.permute(0, 2, 1, 3)
        out = []
        for index, tensor in enumerate([x1, x2]):
            tensor = F.elu(self.conv1(tensor)).view(-1, 9, 655, 9).permute(0, 1, 3, 2)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.padding1(tensor)
            tensor = F.elu(self.conv2(tensor))
            tensor = self.laynorm1(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling1(tensor)
            tensor = self.padding2(tensor)
            tensor = F.elu(self.conv3(tensor))
            tensor = self.laynorm2(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling2(tensor)
            if index == 0:
                out = tensor
            elif index == 1:
                out = torch.cat([out, tensor], dim=1)
        # Dense
        x = out.view(x.shape[0], 1944)
        x = torch.softmax(self.fc1(x), dim=1)
        return x


class BrainGridNet_Raw_Branch1(nn.Module):

    def __init__(self):
        super(BrainGridNet_Raw_Branch1, self).__init__()
        self.conv1 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        self.padding1 = nn.ZeroPad2d((7, 8, 1, 2))
        self.conv2 = nn.Conv2d(9, 36, (4, 16), groups=9)
        self.laynorm1 = nn.LayerNorm([36, 9, 655])
        self.pooling1 = nn.MaxPool2d((3, 6))
        self.padding2 = nn.ZeroPad2d((3, 4, 0, 1))
        self.conv3 = nn.Conv2d(36, 36, (2, 8), groups=36)
        self.laynorm2 = nn.LayerNorm([36, 3, 109])
        self.pooling2 = nn.MaxPool2d((2, 4))
        self.fc1 = nn.Linear(972, 5)  # 1944->972, because only one branch is used

    def forward(self, x):
        x1 = x
        # x2 = x.permute(0, 2, 1, 3)
        out = []
        for index, tensor in enumerate([x1]):  # , x2
            tensor = F.elu(self.conv1(tensor)).view(-1, 9, 655, 9).permute(0, 1, 3, 2)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.padding1(tensor)
            tensor = F.elu(self.conv2(tensor))
            tensor = self.laynorm1(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling1(tensor)
            tensor = self.padding2(tensor)
            tensor = F.elu(self.conv3(tensor))
            tensor = self.laynorm2(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling2(tensor)
            if index == 0:
                out = tensor
            elif index == 1:
                out = torch.cat([out, tensor], dim=1)
        # Dense
        x = out.view(x.shape[0], 972)  # 1944->972, because only one branch is used
        x = torch.softmax(self.fc1(x), dim=1)
        return x


class BrainGridNet_Raw_Branch2(nn.Module):

    def __init__(self):
        super(BrainGridNet_Raw_Branch2, self).__init__()
        self.conv1 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        self.padding1 = nn.ZeroPad2d((7, 8, 1, 2))
        self.conv2 = nn.Conv2d(9, 36, (4, 16), groups=9)
        self.laynorm1 = nn.LayerNorm([36, 9, 655])
        self.pooling1 = nn.MaxPool2d((3, 6))
        self.padding2 = nn.ZeroPad2d((3, 4, 0, 1))
        self.conv3 = nn.Conv2d(36, 36, (2, 8), groups=36)
        self.laynorm2 = nn.LayerNorm([36, 3, 109])
        self.pooling2 = nn.MaxPool2d((2, 4))
        self.fc1 = nn.Linear(972, 5)

    def forward(self, x):
        x2 = x.permute(0, 2, 1, 3)
        out = []
        for index, tensor in enumerate([x2]):  # x1,
            tensor = F.elu(self.conv1(tensor)).view(-1, 9, 655, 9).permute(0, 1, 3, 2)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.padding1(tensor)
            tensor = F.elu(self.conv2(tensor))
            tensor = self.laynorm1(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling1(tensor)
            tensor = self.padding2(tensor)
            tensor = F.elu(self.conv3(tensor))
            tensor = self.laynorm2(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling2(tensor)
            if index == 0:
                out = tensor
            elif index == 1:
                out = torch.cat([out, tensor], dim=1)
        # Dense
        x = out.view(x.shape[0], 972)  # 1944->972, because only one branch is used
        x = torch.softmax(self.fc1(x), dim=1)
        return x


class BrainGridNet_Raw_without_TNet(nn.Module):
    """
    Removes the TimesNet method from the original model. For more details, please refer to Fig. 8 in our paper.
    """

    def __init__(self):
        super(BrainGridNet_Raw_without_TNet, self).__init__()
        self.conv1 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0), groups=9)
        self.padding1 = nn.ZeroPad2d((7, 8, 1, 2))
        self.conv2 = nn.Conv2d(9, 36, (4, 16), groups=9)
        self.laynorm1 = nn.LayerNorm([36, 9, 655])
        self.pooling1 = nn.MaxPool2d((3, 6))
        self.padding2 = nn.ZeroPad2d((3, 4, 0, 1))
        self.conv3 = nn.Conv2d(36, 36, (2, 8), groups=36)
        self.laynorm2 = nn.LayerNorm([36, 3, 109])
        self.pooling2 = nn.MaxPool2d((2, 4))
        self.fc1 = nn.Linear(1944, 5)

    def forward(self, x):
        x1 = x
        x2 = x.permute(0, 2, 1, 3)
        out = []
        for index, tensor in enumerate([x1, x2]):
            tensor = F.elu(self.conv1(tensor))
            tensor = F.dropout(tensor, 0.25)
            tensor = self.padding1(tensor)
            tensor = F.elu(self.conv2(tensor))
            tensor = self.laynorm1(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling1(tensor)
            tensor = self.padding2(tensor)
            tensor = F.elu(self.conv3(tensor))
            tensor = self.laynorm2(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling2(tensor)
            if index == 0:
                out = tensor
            elif index == 1:
                out = torch.cat([out, tensor], dim=1)
        # Dense
        x = out.view(x.shape[0], 1944)
        x = torch.softmax(self.fc1(x), dim=1)
        return x


class BrainGridNet_Raw_without_depthwise(nn.Module):

    def __init__(self):
        super(BrainGridNet_Raw_without_depthwise, self).__init__()
        self.conv1 = nn.Conv2d(9, 9, kernel_size=(9, 1), padding=(4, 0))  # , groups=9
        self.padding1 = nn.ZeroPad2d((7, 8, 1, 2))
        self.conv2 = nn.Conv2d(9, 36, (4, 16))  # , groups=9
        self.laynorm1 = nn.LayerNorm([36, 9, 655])
        self.pooling1 = nn.MaxPool2d((3, 6))
        self.padding2 = nn.ZeroPad2d((3, 4, 0, 1))
        self.conv3 = nn.Conv2d(36, 36, (2, 8))  # , groups=36
        self.laynorm2 = nn.LayerNorm([36, 3, 109])
        self.pooling2 = nn.MaxPool2d((2, 4))
        self.fc1 = nn.Linear(1944, 5)

    def forward(self, x):
        x1 = x
        x2 = x.permute(0, 2, 1, 3)
        out = []
        for index, tensor in enumerate([x1, x2]):
            tensor = F.elu(self.conv1(tensor)).view(-1, 9, 655, 9).permute(0, 1, 3, 2)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.padding1(tensor)
            tensor = F.elu(self.conv2(tensor))
            tensor = self.laynorm1(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling1(tensor)
            tensor = self.padding2(tensor)
            tensor = F.elu(self.conv3(tensor))
            tensor = self.laynorm2(tensor)
            tensor = F.dropout(tensor, 0.25)
            tensor = self.pooling2(tensor)
            if index == 0:
                out = tensor
            elif index == 1:
                out = torch.cat([out, tensor], dim=1)
        # Dense
        x = out.view(x.shape[0], 1944)
        x = torch.softmax(self.fc1(x), dim=1)
        return x


if __name__ == '__main__':
    model = BrainGridNet_PSD32()
    in_data = Variable(torch.Tensor(np.random.rand(1, 9, 9, 32)))
    out = model.forward(in_data)
    # flops = FlopCountAnalysis(model, in_data)
    # print("FLOPs: ", flops.total())
    # print(parameter_count_table(model))
    # flops, params = profile(model, inputs=(in_data,))
    # print('flops', flops)
    # print('params', params)
