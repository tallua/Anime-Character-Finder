import numpy as np
    
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from utils.layers import ConvLayer, ResidualLayer, UpscaleLayer, YoloLayer

class Darknet53(nn.Module):
    def __init__(self, config):
        super(Darknet53, self).__init__()
        self.baseline = nn.Sequential(
            ConvLayer(config['image_size'][2], 32, 3, 1),
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

class YoloNet(nn.Module):
    def __init__(self, config):
        super(YoloNet, self).__init__()
        
        anchor_cnt = len(config['anchors'])
        anchor_per_unit = anchor_cnt // 3
        attrib_count = config['attrib_count']
        
        self.conv3 = nn.Sequential(
            ConvLayer(1024, 512, 1, 1, 0),
            ConvLayer(512, 1024, 3, 1, 1),
            ConvLayer(1024, 512, 1, 1, 0),
            ConvLayer(512, 1024, 3, 1, 1),
            ConvLayer(1024, 512, 1, 1, 0),
        )

        self.detect3 = nn.Sequential(
            ConvLayer(512, 1024, 3, 1, 1),
            ConvLayer(1024, attrib_count * anchor_per_unit, 1, 1, 0),
            YoloLayer(config, config['anchors'][anchor_per_unit * 2 : anchor_cnt], 32)
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
            ConvLayer(512, attrib_count * anchor_per_unit, 1, 1, 0),
            YoloLayer(config, config['anchors'][anchor_per_unit : anchor_per_unit * 2], 16)
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
            ConvLayer(256, attrib_count * anchor_per_unit, 1, 1, 0),
            YoloLayer(config, config['anchors'][0 : anchor_per_unit], 8)
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
        
        assert config['image_size'][0] % 32 is 0, ('image_size[0] should be multiple of 32')
        assert config['image_size'][1] % 32 is 0, ('image_size[1] should be multiple of 32')
        assert len(config['anchors']) % 3 is 0, ('len(anchors) should be multiple of 3')
        assert config['class_count'] >= 0, ('class_count should be equal or above 0')
        
        self.darknet = Darknet53(config)
        self.yolonet = YoloNet(config)

    def forward(self, image):
        out1, out2, out3 = self.darknet(image)
        out1, out2, out3 = self.yolonet(out1, out2, out3)
        return out1, out2, out3



class YoloLoss(object):
    def __init__(self, config):
        self.device = config['device']
        
        self.coef_noobj = torch.tensor(config['coef_noobj']).to(self.device)
        self.coef_coord = torch.tensor(config['coef_coord'] / 
                                       (config['image_size'][0] * config['image_size'][1])).to(self.device) 
        self.iou_threshold = config['iou_threshold']
        
        self.debug_level = config['debug_level']
        
        self.iou_epsilon = torch.tensor(1e-9).to(self.device)
        
    def __call__(self, pred, label, label_len):
        if self.debug_level >= 2:
            print('pred shape: ', pred.shape)
            print('label shape: ', label.shape)
            print('label_len: ', label_len)
            
        # pred = B * P * Attrib
        # label = B * 15 * Attrib
        pred = F.relu(pred)
        label = F.relu(label)
        
        if self.debug_level >= 3:
            print('pred : ', pred)
            print('label_len : ', label_len)
            print('label : ', label)
            
        # iou = B * P * 15
        # obj_mask = B * P * 15
        iou = self.batch_iou(pred, label)
        obj_mask = self.batch_obj_mask(iou, label_len)
        if self.debug_level >= 3:
            print('iou : ', iou)
        if self.debug_level >= 2:
            print('obj_mask.shape : ', obj_mask.shape)
        
        # objectness loss = B * P * 15
        obj_loss = self.batch_obj_loss(obj_mask, pred, label_len)
        if self.debug_level >= 2:
            print('obj_loss : ', obj_loss)
        
        # coord loss = B * P * 15
        coord_loss = self.batch_coord_loss(obj_mask, pred, label)
        if self.debug_level >= 2:
            print('coord_loss : ', coord_loss)
        
        # classfication loss = B * P * 15
        #class_loss = obj_mask * self.batch_class_loss(pred, label)
        
        total_loss = torch.sum(obj_loss + coord_loss)
        if self.debug_level >= 1:
            print('total_loss : ', total_loss)
        
        return total_loss
        
        
        ###
        num_batch = pred.shape[0]
        
        total_loss = torch.tensor([0.0], device=self.device)
        
        for batch_idx in range(num_batch):
            
            iou = self.calc_iou(pred[batch_idx], label[batch_idx])
            responsibile = self.calc_responsibile(iou)
            
            # coord loss
            coord_loss = self.calc_coord_loss(pred[batch_idx], label[batch_idx], responsibile)
            mean_coord_loss = torch.mean(coord_loss) * self.coef_coord
            
            if self.debug_level >= 2:
                print('mean_coord_loss', mean_coord_loss)
            total_loss += mean_coord_loss
            
            # confidence loss
            conf_loss = self.calc_confidence_loss(pred[batch_idx], iou, responsibile)
            mean_conf_loss = torch.mean(conf_loss)
            if self.debug_level >= 2:
                print('mean_coord_loss', mean_coord_loss)
            total_loss += mean_conf_loss
            
            # classification loss
            # TODO
            # not implemented (we firstly aims to classify bbox of face only)
            
            
            if self.debug_level >= 2:
                print('iou : ', iou)
                print('iou type: ', iou.type())
                print('iou shape: ', iou.shape)
                print('responsibile : ', responsibile)
                print('coord_loss : ', coord_loss)
                print('coord_loss type : ', coord_loss.type())
                print('conf_loss : ', conf_loss)
                print('conf_loss type : ', conf_loss.type())
        if self.debug_level >= 1:
            print('loss : ', total_loss / num_batch * self.coef_total)
        
        return total_loss / num_batch
    
    # why does this works?
    def batch_iou(self, pred, label):
        x1 = label[..., 0]
        y1 = label[..., 1]
        w1 = label[..., 2]
        h1 = label[..., 3]

        x2 = pred[..., 0]
        y2 = pred[..., 1]
        w2 = pred[..., 2]
        h2 = pred[..., 3]

        area1 = w1 * h1
        area2 = w2 * h2

        x1 = x1 - w1 / 2
        y1 = y1 - h1 / 2
        x2 = x2 - w2 / 2
        y2 = y2 - h2 / 2
        right1 = (x1 + w1).unsqueeze(2)
        right2 = (x2 + w2).unsqueeze(1)
        top1 = (y1 + h1).unsqueeze(2)
        top2 = (y2 + h2).unsqueeze(1)
        left1 = x1.unsqueeze(2)
        left2 = x2.unsqueeze(1)
        bottom1 = y1.unsqueeze(2)
        bottom2 = y2.unsqueeze(1)
        
        
        w_intersect = (torch.min(right1, right2) - torch.max(left1, left2)).clamp(min=0)
        h_intersect = (torch.min(top1, top2) - torch.max(bottom1, bottom2)).clamp(min=0)
        area_intersect = h_intersect * w_intersect

        iou_ = area_intersect / (area1.unsqueeze(2) + area2.unsqueeze(1) - area_intersect + self.iou_epsilon)

        return iou_
        
    def batch_obj_mask(self, iou, label_len):
        max_iou, max_iou_indx = torch.max(iou, 2)
        if self.debug_level >= 2:
            print('max_iou ', max_iou)
        
        obj_mask = torch.where(iou > max_iou.unsqueeze(2) * self.iou_threshold, 
                               torch.ones_like(iou), torch.zeros_like(iou))
        
        if self.debug_level >= 2:
            print('nonzero ', torch.nonzero(obj_mask).shape[0])
        
        return obj_mask
    
    def batch_obj_loss(self, obj_mask, pred, label_len):
        coef_mask = torch.where(obj_mask == 1, 
                                torch.ones_like(obj_mask), torch.ones_like(obj_mask) * self.coef_noobj)
        
        conf = torch.transpose(pred[..., 4].clone().repeat(obj_mask.shape[1], 1, 1), 0, 1)
        
        obj_loss_all = coef_mask * F.mse_loss(obj_mask, conf, reduction='none')
        
        obj_loss = torch.tensor(0.0, device = self.device)
        for indx in range(0, label_len.shape[0]):
            obj_loss += torch.sum(obj_loss_all[indx][0:label_len[indx]])
        
        if self.debug_level >= 2:
            print('max conf : ', torch.max(obj_loss_all))
        if self.debug_level >= 4:
            print('obj_loss.shape', obj_loss.shape)
        
        return obj_loss
        
    def batch_coord_loss(self, obj_mask, pred, label):
        x1 = label[..., 0].repeat(pred.shape[1], 1, 1).permute(1, 2, 0)
        y1 = label[..., 1].repeat(pred.shape[1], 1, 1).permute(1, 2, 0)
        w1 = label[..., 2].repeat(pred.shape[1], 1, 1).permute(1, 2, 0)
        h1 = label[..., 3].repeat(pred.shape[1], 1, 1).permute(1, 2, 0)
        
        x2 = torch.transpose(pred[..., 0].repeat(label.shape[1], 1, 1), 0, 1)
        y2 = torch.transpose(pred[..., 1].repeat(label.shape[1], 1, 1), 0, 1)
        w2 = torch.transpose(pred[..., 2].repeat(label.shape[1], 1, 1), 0, 1)
        h2 = torch.transpose(pred[..., 3].repeat(label.shape[1], 1, 1), 0, 1)
        
        x_loss = self.coef_coord * obj_mask * F.mse_loss(x1, x2, reduction='none')
        y_loss = self.coef_coord * obj_mask * F.mse_loss(y1, y2, reduction='none')
        w_loss = self.coef_coord * obj_mask * F.mse_loss(torch.sqrt(w1), torch.sqrt(w2), reduction='none')
        h_loss = self.coef_coord * obj_mask * F.mse_loss(torch.sqrt(h1), torch.sqrt(h2), reduction='none')
        
        coord_loss = x_loss + y_loss + w_loss + h_loss
        coord_loss = torch.sum(coord_loss)
        if self.debug_level >= 4:
            print('coord_loss.shape', coord_loss.shape)
        
        return coord_loss
    

    


    
class PostProcessor(object):
    
    def iou(self, bbox1, bbox2):
        w1 = bbox1[2]
        w2 = bbox1[2]
        h1 = bbox1[3]
        h2 = bbox1[3]
    
        left1 = bbox1[0] - w1 / 2
        left2 = bbox2[0] - w2 / 2
        right1 = bbox1[0] + w1 / 2
        right2 = bbox2[0] + w2 / 2
        top1 = bbox1[1] + h1 / 2
        top2 = bbox2[1] + h2 / 2
        bottom1 = bbox1[1] - h1 / 2
        bottom2 = bbox2[1] - h2 / 2
    
        area1 = w1 * h1
        area2 = w2 * h2
    
        w_intersect = min(right1, right2) - max(left1, left2)
        h_intersect = min(top1, top2) - max(bottom1, bottom2)
        area_intersect = h_intersect * w_intersect

        iou_ = area_intersect / (area1 + area2 - area_intersect + 1e-9)
        return iou_
        
    # non-maximum suppression
    def NMS(self, prediction, context):
        batch_count = prediction.shape[0]
        
        result = []
        for indx in range(0, batch_count):
            pred = prediction[indx].detach().numpy()
        
            above_thres = pred[np.where(pred[:, 4] > context['post_mAp'])]
            pred_sorted = np.flip(np.argsort(above_thres[:, 4]))
        
            max_list = []
            for pred_it in pred_sorted:
                discard = False
                for max_it in range(0, len(max_list)):
                    if self.iou(above_thres[pred_it], above_thres[max_it]) > context['post_iou_threshold']:
                        discard = True
                        break
                    
                if discard is False:
                    max_list.append(pred_it)
                    
            max_boxes = [above_thres[max_it] for max_it in max_list]
            result.append(max_boxes)
            
        return result
        
    # group-average suppression
    def GAS(self, prediction, context):
        batch_count = prediction.shape[0]
        
        result = []
        for indx in range(0, batch_count):
            pred = prediction[indx].detach().numpy()
        
            above_thres = pred[np.where(pred[:, 4] > context['post_mAp'])]
            pred_sorted = np.flip(np.argsort(above_thres[:, 4]))
            
            groups = []
            for pred_it in pred_sorted:
                new_region = True
                for g_it in range(0, len(groups)):
                    if self.iou(above_thres[pred_it], np.mean(above_thres[groups[g_it]], axis = 0)) > context['post_iou_threshold']:
                        groups[g_it].append(pred_it)
                        new_region = False
                    
                if new_region is True:
                    groups.append([pred_it])
                    
            avg_boxes = [np.mean(above_thres[g_it], axis = 0) for g_it in groups]
            result.append(avg_boxes)
            
        return result
    


