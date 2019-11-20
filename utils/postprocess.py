import numpy as np

import PIL.Image as Image

class PostProcessor(object):
    
    def resize(self, from_size, to_size, prediction):
        x_rate = to_size[0] / from_size[0]
        y_rate = to_size[1] / from_size[1]
        
        prediction['pred'] = np.array([pred * [x_rate, y_rate, x_rate, y_rate, 1] for pred in prediction['pred']])
        prediction['label'] = np.array([label * [x_rate, y_rate, x_rate, y_rate, 1] for label in prediction['label']])
        
        image = Image.fromarray((prediction['image'] * 255).astype('uint8'))
        image = image.resize(to_size, Image.BILINEAR)
        prediction['image'] = np.array(image)
        
        return prediction
    
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
        
        if h_intersect < 0 or w_intersect < 0:
            return 0

        iou_ = area_intersect / (area1 + area2 - area_intersect + 1e-9)
        return iou_
        
    # all bbox above conf_threshold
    def ABOVE(self, prediction, context):
        above_thres = prediction[np.where(prediction[:, 4] > context['post_conf_threshold'])]
        return above_thres
        
        
    # non-maximum suppression
    def NMS(self, prediction, context):
        above_thres = prediction[np.where(prediction[:, 4] > context['post_conf_threshold'])]
        pred_sorted = np.flip(np.argsort(above_thres[:, 4]))
            
        pred_result = []
        for p0 in pred_sorted:
            discard = False
            for p1 in pred_result:
                if self.iou(above_thres[p0], above_thres[p1]) > context['post_iou_threshold']:
                    discard = True
                    break
            if discard is False:
                pred_result.append(p0)
                
        pred_result = np.array(above_thres[pred_result])
        
        return pred_result
    
    # custom 1
    def CUSTOM1(self, prediction, context):
        above_thres = prediction[np.where(prediction[:, 4] > context['post_conf_threshold'])]
        pred_sorted = np.flip(np.argsort(above_thres[:, 4]))
            
        pred_result = []
        for p0 in pred_sorted:
            new_group = True
            max_matching_group = 0
            max_iou = 0
                    
            for g1 in range(0, len(pred_result)):
                iou_match = self.iou(above_thres[p0], np.mean(pred_result[g1], axis = 0))
                if iou_match > context['post_iou_threshold']:
                    new_group = False
                    if max_iou < iou_match:
                        max_iou = iou_match
                        max_matching_group = g1
                                
            if new_group is True:
                pred_result.append([above_thres[p0]])
            else:
                pred_result[max_matching_group].append(above_thres[p0])
                        
        pred_result = np.array([np.mean(pred_group, axis = 0) for pred_group in pred_result])
        return pred_result
        
    def CUSTOM2(self, prediction, context):
        above_thres = np.copy(prediction[np.where(prediction[:, 4] > context['post_conf_threshold'])])
        pred_sorted = above_thres[np.flip(np.argsort(above_thres[:, 4]))]
            
        # merge with max iou until converge
        pred_result = []
        converge = False
        while converge is False:
            if len(pred_sorted) is 0:
                converge = True
                break
                        
            max_iou = 0
            max_indx = 0
                    
            p0 = pred_sorted[0]
            for p_indx in range(1, len(pred_sorted)):
                iou_match = self.iou(p0, pred_sorted[p_indx])
                if iou_match > context['post_iou_threshold'] and iou_match > max_iou:
                    max_iou = iou_match
                    max_indx = p_indx
                    
            if max_indx is not 0:
                weight_0 = pred_sorted[0][4]
                weight_1 = pred_sorted[max_indx][4]
                weight_sum = weight_0 + weight_1
                        
                avg  = (pred_sorted[0] * weight_0 / weight_sum) + (pred_sorted[max_indx] * weight_1 / weight_sum)
                        
                pred_sorted = np.delete(pred_sorted, max_indx, 0)
                pred_sorted = np.delete(pred_sorted, 0, 0)
                pred_sorted = np.append(pred_sorted, [avg], 0)
            else:
                pred_result.append(p0)
                pred_sorted = np.delete(pred_sorted, 0, 0)
                        
            if len(pred_sorted) is 0:
                converge = True
            else:
                pred_sorted = pred_sorted[np.flip(np.argsort(pred_sorted[:, 4]))]
                
        return pred_result

    def calcAccuracyMap(self, truth, truth_len, pred, context):
        
        check_arr = np.zeros(truth_len)
        check_fp = 0

        for p in pred:
            max_indx = -1
            max_iou = 0.01

            for i in range(0, truth_len):
                iou_val = self.iou(p, truth[i])
                if max_iou < iou_val:
                    max_iou = iou_val
                    max_indx = i
            
            if max_indx is -1:
                check_fp = check_fp + 1
            else:
                if max_iou > context['acc_iou_threshold']:
                    check_arr[max_indx] = check_arr[max_indx] + 1
                else:
                    check_fp = check_fp + 1
    
        result = {}
        result['count'] = truth_len.item()
        result['true positive'] = np.argwhere(check_arr != 0).size
        result['false negative'] = np.argwhere(check_arr == 0).size
        result['false positive'] = check_fp
        result['duplicate'] = np.argwhere(check_arr > 1).size 

        return result





