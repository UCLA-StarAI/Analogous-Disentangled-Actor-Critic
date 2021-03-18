import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, input_c, hidden_c, output_c, block_num, pool = "none"):
        super(DenseBlock, self).__init__()

        self.input_c = input_c
        self.hidden_c = hidden_c
        self.output_c = output_c
        self.block_num = block_num

        self.convBlocks = []

        for block_idx in range(block_num):
            unit = nn.Sequential(
                    nn.Conv2d(input_c + block_idx * hidden_c, hidden_c, 1, 1, 0, bias = True),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(hidden_c, hidden_c, 3, 1, 1, bias = True),
                    nn.ReLU(inplace = True)
                    )
            self.add_module("block_unit_" + str(block_idx), unit)
            self.convBlocks.append(unit)

        if pool == "none":
            self.endingSeq = nn.Sequential(
                                nn.Conv2d(input_c + block_num * hidden_c, output_c, 1, 1, 0, bias = True),
                                nn.BatchNorm2d(output_c),
                                nn.ReLU(inplace = True)
                                )
        elif pool == "aveg":
            self.endingSeq = nn.Sequential(
                                nn.Conv2d(input_c + block_num * hidden_c, output_c, 1, 1, 0, bias = True),
                                nn.BatchNorm2d(output_c),
                                nn.ReLU(inplace = True),
                                nn.AvgPool2d(2)
                                )
        elif pool == "aveg":
            self.endingSeq = nn.Sequential(
                                nn.Conv2d(input_c + block_num * hidden_c, output_c, 1, 1, 0, bias = True),
                                nn.BatchNorm2d(output_c),
                                nn.ReLU(inplace = True),
                                nn.MaxPool2d(2)
                                )
        else:
            raise NotImplementedError("Unexpected arg for pool")

    def forward(self, x):
        maps = x

        for convBlock in self.convBlocks:
            h = convBlock(maps)
            maps = torch.cat((maps, h), dim = 1)

        y = self.endingSeq(maps)

        return y

class DenseNet(nn.Module):
    def __init__(self, input_c, output_vec_len = 128):
        super(DenseNet, self).__init__()

        self.DenseBlock1 = DenseBlock(input_c = input_c, hidden_c = 16, output_c = 32, block_num = 4, pool = "none")

        self.DenseBlock2 = DenseBlock(input_c = 32, hidden_c = 16, output_c = 32, block_num = 6, pool = "aveg")

        self.DenseBlock3 = DenseBlock(input_c = 32, hidden_c = 32, output_c = 64, block_num = 4, pool = "none")

        self.AdaAvgPool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
                        nn.Linear(64, output_vec_len),
                        nn.ReLU(inplace = True)
                        )

    def forward(self, x):
        x = self.DenseBlock1(x)
        x = self.DenseBlock2(x)
        x = self.DenseBlock3(x)

        x = self.AdaAvgPool(x).view(x.size(0), -1)
        h = self.linear(x)

        return h
