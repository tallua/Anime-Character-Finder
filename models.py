 
import torch as torch
import torch.nn as nn

from utils.layers import ConvLayer, ResidualLayer, UpscaleLayer, YoloLayer

class Darknet53(nn.Module):
    def __init__(self, config):
        super(Darknet53, self).__init__()
        self.baseline = nn.Sequential(
            ConvLayer(config['channel'], 32, 3, 1),
            ConvLayer(32, 64, 3, 2),

            ResidualLayer(64),

            ConvLayer(64, 128, 3, 2),

            ResidualLayer(128),
            ResidualLayer(128)
        )

        self.route1 = nn.Sequential(
            ConvLayer(128, 256, 3, 2),

            ## 8x
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256)
        )

        self.route2 = nn.Sequential(
            ConvLayer(256, 512, 3, 2),

            ## 8x
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512)
        )

        self.route3 = nn.Sequential(
            ConvLayer(512, 1024, 3, 2),

            ## 4x
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

    def forward(self, image):
        image = self.baseline(image)
        route1 = self.route1(image)
        route2 = self.route2(route1)
        route3 = self.route3(route2)

        return route1, route2, route3

### from https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch/blob/release/src/model.py
class YoloNet(nn.Module):
    def __init__(self, config):
        super(YoloNet, self).__init__()
        
        anch_cnt = len(config['anchors'])
        
        self.conv3 = nn.Sequential(
            ConvLayer(1024, 512, 1, 1, 0),
            ConvLayer(512, 1024, 3, 1, 1),
            ConvLayer(1024, 512, 1, 1, 0),
            ConvLayer(512, 1024, 3, 1, 1),
            ConvLayer(1024, 512, 1, 1, 0),
        )

        self.detect3 = nn.Sequential(
            ConvLayer(512, 1024, 3, 1, 1),
            ConvLayer(1024, config['attribs'] * anch_cnt // 3, 1, 1, 0),
            YoloLayer(config, config['anchors'][anch_cnt // 3 * 2 : anch_cnt], 32)
        )

        self.conv2_1 = nn.Sequential(
            ConvLayer(512, 256, 1, 1, 0),
            UpscaleLayer(2)
        )

        self.conv2_2 = nn.Sequential(
            ConvLayer(768, 256, 1, 1, 0),
            ConvLayer(256, 512, 3, 1, 1),
            ConvLayer(512, 256, 1, 1, 0),
            ConvLayer(256, 512, 3, 1, 1),
            ConvLayer(512, 256, 1, 1, 0)
        )

        self.detect2 = nn.Sequential(
            ConvLayer(256, 512, 3, 1, 1),
            ConvLayer(512, config['attribs'] * anch_cnt // 3, 1, 1, 0),
            YoloLayer(config, config['anchors'][anch_cnt // 3 * 1 : anch_cnt // 3 * 2], 16)
        )

        self.conv1_1 = nn.Sequential(
            ConvLayer(256, 128, 1, 1, 0),
            UpscaleLayer(2)
        )

        self.conv1_2 = nn.Sequential(
            ConvLayer(384, 128, 1, 1, 0),
            ConvLayer(128, 256, 3, 1, 1),
            ConvLayer(256, 128, 1, 1, 0),
            ConvLayer(128, 256, 3, 1, 1),
            ConvLayer(256, 128, 1, 1, 0)
        )

        self.detect1 = nn.Sequential(
            ConvLayer(128, 256, 3, 1, 1),
            ConvLayer(256, config['attribs'] * anch_cnt // 3, 1, 1, 0),
            YoloLayer(config, config['anchors'][0 : anch_cnt // 3 * 1], 8)
        )

    def forward(self, in1, in2, in3):
        #     in3
        # in2  |
        #     \|\ 
        # in1  |  out3
        #     \|\
        #      |  out2
        #     out1
        tmp = self.conv3(in3)
        out3 = self.detect3(tmp)
        tmp = self.conv2_1(tmp)
        tmp = torch.cat((tmp, in2), 1)
        tmp = self.conv2_2(tmp)
        out2 = self.detect2(tmp)
        tmp = self.conv1_1(tmp)
        tmp = torch.cat((tmp, in1), 1)
        tmp = self.conv1_2(tmp)
        out1 = self.detect1(tmp)
        
        return out1, out2, out3

class YoloV3(nn.Module):
    def __init__(self, config):
        super(YoloV3, self).__init__()
        
        assert config['size'][0] % 32 is 0
        assert config['size'][1] % 32 is 0 
        
        self.darknet = Darknet53(config)
        self.yolonet = YoloNet(config)

    def forward(self, image):
        out1, out2, out3 = self.darknet(image)
        out1, out2, out3 = self.yolonet(out1, out2, out3)
        return out1, out2, out3







