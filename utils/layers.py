
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, filter_in, filter_out, kernel, stride = 1, padding = 1):
        super(ConvLayer, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(filter_in, filter_out, kernel_size = kernel, stride = stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features = filter_out),
            nn.LeakyReLU()
        )

    def forward(self, input):
        return self.body(input)


class ResidualLayer(nn.Module):
    def __init__(self, filter):
        super(ResidualLayer, self).__init__()
        self.block = nn.Sequential(
            ConvLayer(filter, filter // 2, 1, 1, 0),
            ConvLayer(filter // 2, filter, 3, 1, 1),
        )

    def forward(self, input):
        tmp = self.block(input)
        return tmp + input


class UpscaleLayer(nn.Module):
    def __init__(self, scale = 2, mode = 'nearest'):
        super(UpscaleLayer, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, input):
        return F.interpolate(input, scale_factor = self.scale, mode = self.mode)


class YoloLayer(nn.Module):
    def __init__(self, config, anchors, stride):
        super(YoloLayer, self).__init__()
        
        self.num_grid = config['image_size'][0] // stride
        
        self.stride = stride
        self.anchors = torch.tensor(anchors).to(config['device'], dtype = config['dtype'])
        self.anchor_cnt = len(anchors)
        self.attrib_cnt = config['attrib_count']
        
        grid_tensor = torch.arange(self.num_grid).repeat(self.num_grid, 1) + 0.5
        grid_tensor = grid_tensor.to(config['device'], dtype = config['dtype'])
        self.grid_x = grid_tensor.view([1, 1, self.num_grid, self.num_grid])
        self.grid_y = grid_tensor.t().view([1, 1, self.num_grid, self.num_grid])
        self.anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1)).float()
        self.anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1)).float()
        

    def forward(self, x):
        num_batch = x.size(0)
        
        prediction_raw = x.view(num_batch,
                                    self.anchor_cnt,
                                    self.attrib_cnt,
                                    self.num_grid,
                                    self.num_grid).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + self.grid_x) * self.stride # Center x
        y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + self.grid_y) * self.stride  # Center y
        w_pred = torch.exp(torch.sigmoid(prediction_raw[..., 2])) * self.anchor_w  # Width
        h_pred = torch.exp(torch.sigmoid(prediction_raw[..., 3])) * self.anchor_h  # Height
        
        bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4).view((num_batch, -1, 4)) #cxcywh
        conf_pred = torch.sigmoid(prediction_raw[..., 4] * 10).view(num_batch, -1, 1)  # Conf
        output = torch.cat((bbox_pred, conf_pred), -1)
        
        if self.attrib_cnt > 5:
            cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, 1)  # Cls pred one-hot.
            output = torch.cat((output, cls_pred), -1)
        return output

