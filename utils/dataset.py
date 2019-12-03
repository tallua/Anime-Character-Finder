from os import mkdir
from os import listdir
from pathlib import Path
from fnmatch import fnmatch

import json

import torch
from torch.utils.data import Dataset

import copy

import PIL.Image
import numpy as np

import random


###
# Sequence to process data
###
class ProcessSequence(object):
    def __init__(self, process = []):
        self.process_list = process

    def add(self, process):
        self.process_list.append(process)

    def __call__(self, params):
        for process in self.process_list:
            process(params)
                
###
# Process components
###    
class ImageLoadProcess(object):
    def __init__(self, key_image_path, key_image):
        self.key_image_path = key_image_path
        self.key_image = key_image
    
    def __call__(self, result):
        if self.key_image_path not in result:
            return
        
        try:
            image_data = PIL.Image.open(result[self.key_image_path])
            result[self.key_image] = PIL.Image.new("RGB", (image_data.size[0], image_data.size[1]))
            result[self.key_image].paste(image_data, (0, 0))
            #result[self.key_image] = result[self.key_image].crop(result[self.key_image].getbbox())
        except IOError:
            print('cannot load image : ', result[self.key_image_path])

class LabelLoadProcess(object):
    def __init__(self, key_label_path, key_label, key_label_len, max_label_len = 15):
        self.key_label_path = key_label_path
        self.key_label = key_label
        self.key_label_len = key_label_len
        self.max_label_len = max_label_len
    
    def __call__(self, result):
        if self.key_label_path not in result:
            return
        
        with open(result[self.key_label_path], 'r') as label_file:
            label_lines = label_file.readlines()
            labels = np.array([label_line.split(',') for label_line in label_lines]).astype(np.float)
            labels[:, 0:4] = np.array([[(label[0] + label[2]) / 2, (label[1] + label[3]) / 2,
                            label[2] - label[0], label[3] - label[1]] for label in labels[:, 0:4]])
            labels = labels.astype(np.float)

            result[self.key_label_len] = labels.shape[0]

            if labels.shape[0] > self.max_label_len:
                labels = labels[0:self.max_label_len]
            elif labels.shape[0] < self.max_label_len:
                labels = np.concatenate((labels, np.zeros((self.max_label_len - labels.shape[0], labels.shape[1]))), axis = 0)

            result[self.key_label] = labels

class DataResizeProcess(object):
    def __init__(self, key_image, key_label = None, keep_ratio = True, from_size = None, 
        to_size = (608, 608), opt = PIL.Image.NEAREST):

        self.key_image = key_image
        self.key_label = key_label
        self.keep_ratio = keep_ratio
        self.from_size = from_size
        self.to_size = to_size
        self.opt = opt
    
    def __call__(self, result):
        if self.key_image not in result:
            return

        if self.from_size is not None:
            from_size = self.from_size
        else:
            from_size = result[self.key_image].size
        to_size = self.to_size

        if self.keep_ratio is True:
            min_rate = min(to_size[0] / from_size[0], to_size[1] / from_size[1])
            to_size = (int(from_size[0] * min_rate), int(from_size[1] * min_rate))
            
        result[self.key_image] = result[self.key_image].resize(to_size, self.opt)

        if self.key_label is not None:
            x_rate = to_size[0] / from_size[0]
            y_rate = to_size[1] / from_size[1]
            result[self.key_label][:, 0:4] = np.array([
                [label[0] * x_rate, label[1] * y_rate, label[2] * x_rate, label[3] * y_rate]
                for label in result[self.key_label][:, 0:4]])
    
class DataCropProcess(object):
    def __init__(self, key_image, key_label = None, key_label_len = None, key_bbox = None):
        self.key_image = key_image
        self.key_label = key_label
        self.key_label_len = key_label_len
        self.key_bbox = key_bbox
        
    def __call__(self, result):
        
        result[self.key_image] = result[self.key_image].crop(tuple(result[self.key_bbox]))
        
        if self.key_label is not None:
            offset = result[self.key_bbox]
            offset = np.array([offset[0], offset[1], offset[0], offset[1]])
            w = offset[2] - offset[0]
            h = offset[3] - offset[1]
            
            for i in range(0, result[self.key_label_len]):
                result[self.key_label][i, :4] -= offset
            
            new_labels_tmp = []
            
            for label in result[self.key_label]:
                if label[0] < 0 or label[1] < 0 or label[2] < 0 or label[3] < 0:
                    continue
                    
                if label[0] > w or label[1] > h or label[2] > w or label[3] > h:
                    continue
                    
                new_labels_tmp.append(label)
            
            result[self.key_label] = np.array(new_labels_tmp)
            result[self.key_label_len] = result[self.key_label].shape[0]
        return
    

class RandomDataCropProcess(object):
    def __init__(self, key_image, key_label, min_size = None, ratio_threshold = 0.3, tries = 10):
        self.key_image = key_image
        self.key_label = key_label
        self.min_size = min_size
        self.ratio_threshold = ratio_threshold
        self.tries = tries
        
    def __call__(self, result):
        if (self.key_image not in result) or (self.key_label not in result):
            return
        
        orig_w, orig_h = result[self.key_image].size
        orig_faces = result[self.key_label]
            
        if self.min_size is not None:
            min_w, min_h = min_size
        else:
            min_w = orig_w // 3
            min_h = orig_h // 3
        max_w = orig_w
        max_h = orig_h
            
        for _ in range(0, self.tries) :
            # random select area
            new_w = random.randrange(min_w, max_w, 1)
            new_h = random.randrange(min_h, max_h, 1)
            new_x1 = random.randrange(0, max_w - new_w, 1)
            new_x2 = new_x1 + new_w
            new_y1 = random.randrange(0, max_h - new_h, 1)
            new_y2 = new_y1 + new_h

            # skip area if simillar to original image
            if abs(1 - (new_w / orig_w)) < self.ratio_threshold and abs(1 - (new_h / orig_h)) < self.ratio_threshold:
                continue
            
            # crop labels
            new_faces = orig_faces.copy()
            new_faces[:, 0:4] = np.array([[face[0] - new_x1, face[1] - new_y1, face[2] - new_x1, face[3] - new_y1] 
                                 for face in new_faces[:, 0:4]])
            
            # check collision with boundary
            collide = False
            for new_face in new_faces:
                if new_face[0] * new_face[2] < 0:
                    collide = True
                    break
                if (new_face[0] - new_w) * (new_face[2] - new_w) < 0:
                    collide = True
                    break
                if new_face[1] * new_face[3] < 0:
                    collide = True
                    break
                if (new_face[1] - new_h) * (new_face[3] - new_h) < 0:
                    collide = True
                    break
                    
            if collide is True :
                continue
                    
            if len(new_faces) == 0:
                continue
                
            # end if success
            try:
                result[self.key_image] = result[self.key_image].crop((new_x1, new_y1, new_x2, new_y2))
                result[self.key_label] = new_faces
            except ValueError:
                print('cannot random crop : ', result['title'])
                print('    bbox :', (new_x1, new_y1, new_x2, new_y2))
                
                # fail mark
                del result[self.key_image]

            return
        
        # fail mark
        del result[self.key_image]
        

class DataPaddingProcess(object):
    def __init__(self, key_image, target_size):
        self.key_image = key_image
        self.size = target_size

    def __call__(self, result):
        if self.key_image not in result:
            return

        #result[key_image] = result[key_image].crop(result[key_image].getbbox())

        new_image = PIL.Image.new("RGB", (self.size[0], self.size[1]))
        new_image.paste(result[self.key_image], (0, 0))

        result[self.key_image] = new_image
    
class DataToTensorProcess(object):
    def __init__(self, key_image, key_label = None, key_label_len = None):
        self.key_image = key_image
        self.key_label = key_label
        self.key_label_len = key_label_len
        
    def __call__(self, result):
        if self.key_image not in result:
            return

        result[self.key_image] = np.array(result[self.key_image]) / 255
        result[self.key_image] = torch.from_numpy(result[self.key_image])
        result[self.key_image] = result[self.key_image].permute(2, 0, 1)[0:3]
        
        if (self.key_label not in result) or (self.key_label_len not in result):
            return

        result[self.key_label] = torch.from_numpy(result[self.key_label])
        result[self.key_label_len] = torch.tensor(result[self.key_label_len])
        
class DataValidateProcess(object):
    def __init__(self, key_image, key_label, key_label_len, key_index, threshold_size = (20, 20)):
        self.key_image = key_image
        self.key_label = key_label
        self.key_label_len = key_label_len
        self.key_index = key_index
        
        self.threshold_size = threshold_size
    
    
    def __call__(self, result):
        if (self.key_image not in result) or (self.key_label not in result):
            if self.key_image in result:
                del result[self.key_image]
            if self.key_label in result:
                del result[self.key_label]
            if self.key_label_len in result:
                del result[self.key_label_len]
            if self.key_index in result:
                del result[self.key_index]
            return
        
        
        w, h = result[self.key_image].size
        
        new_faces = []
        for face in result[self.key_label]:
            if face[0] < 0 or face[1] < 0 or face[2] < 0 or face[3] < 0:
                continue
            if face[0] >= w or face[1] >= h or face[2] >= w or face[3] >= h:
                continue
            if (face[2] - face[0] < self.threshold_size[0]) or (face[3] - face[1] < self.threshold_size[1]):
                continue
            
            new_faces.append(face)
            
        if len(new_faces) == 0:
            del result[self.key_image]
            del result[self.key_label]
            if self.key_label_len in result:
                del result[self.key_label_len]
            if self.key_index in result:
                del result[self.key_index]
                
        result[self.key_label] = np.array(new_faces)
        result[self.key_label_len] = len(new_faces)
        return
        

class KeyCopyProcess(object):
    def __init__(self, key_from, key_to):
        self.key_from = key_from
        self.key_to = key_to
    
    def __call__(self, result):
        if self.key_from not in result:
            return

        if 'copy' in dir(result[self.key_from]):
            result[self.key_to] = result[self.key_from].copy()
        else:
            result[self.key_to] = copy.deepcopy(result[self.key_from])


class ImageSaveProcess(object):
    def __init__(self, key_image, key_save_path):
        self.key_image = key_image
        self.key_save_path = key_save_path

    def __call__(self, result):
        if self.key_image not in result:
            return

        result[self.key_image].save(result[self.key_save_path], 'PNG')

class LabelSaveProcess(object):
    def __init__(self, key_label, key_save_path):
        self.key_label = key_label
        self.key_save_path = key_save_path
    
    def __call__(self, result):
        if self.key_label not in result:
            return
            
        with open(result[self.key_save_path], 'w+') as labelfile:
            for bbox in result[self.key_label]:
                first = True
                for val in bbox:
                    if first is False:
                        labelfile.write(',')
                    labelfile.write('%.4f' % val)
                    first = False
                labelfile.write('\n')

class IndexSaveProcess(object):
    def __init__(self, key_index, key_save_path):
        self.key_index = key_index
        self.key_save_path = key_save_path
    
    def __call__(self, result):
        if self.key_index not in result:
            return
    
        with open(result[self.key_save_path], 'a+') as indexfile:
            indexfile.write(result[self.key_index] + '\n')

    
###
# Predfined Datasets
###
class ImagePreProcessor(object):
    def __init__(self, target_size):
        self.sequence = ProcessSequence([
            ImageLoadProcess('background_path', 'image_og'),

            KeyCopyProcess('image_og', 'image_r1'),
            KeyCopyProcess('image_og', 'image_r2'),
            KeyCopyProcess('image_og', 'image_r3'),
            #KeyCopyProcess('image_og', 'image_r4'),
            KeyCopyProcess('label_og', 'label_r1'),
            KeyCopyProcess('label_og', 'label_r2'),
            KeyCopyProcess('label_og', 'label_r3'),
            #KeyCopyProcess('label_og', 'label_r4'),
            KeyCopyProcess('label_len_og', 'label_len_r1'),
            KeyCopyProcess('label_len_og', 'label_len_r2'),
            KeyCopyProcess('label_len_og', 'label_len_r3'),
            #KeyCopyProcess('label_len_og', 'label_len_r4'),

            RandomDataCropProcess('image_r1', 'label_r1', tries = 15),
            RandomDataCropProcess('image_r2', 'label_r2', tries = 15),
            RandomDataCropProcess('image_r3', 'label_r3', tries = 15),
            #RandomDataCropProcess('image_r4', 'label_r4', tries = 15),

            DataResizeProcess('image_og', 'label_og', keep_ratio = True, to_size = target_size),
            DataResizeProcess('image_r1', 'label_r1', keep_ratio = True, to_size = target_size),
            DataResizeProcess('image_r2', 'label_r2', keep_ratio = True, to_size = target_size),
            DataResizeProcess('image_r3', 'label_r3', keep_ratio = True, to_size = target_size),
            #DataResizeProcess('image_r4', 'label_r4', keep_ratio = True, to_size = target_size),
            
            DataValidateProcess('image_og', 'label_og', 'label_len_og', 'index_og'),
            DataValidateProcess('image_r1', 'label_r1', 'label_len_r1', 'index_r1'),
            DataValidateProcess('image_r2', 'label_r2', 'label_len_r2', 'index_r2'),
            DataValidateProcess('image_r3', 'label_r3', 'label_len_r3', 'index_r3'),
            #DataValidateProcess('image_r4', 'label_r4', 'label_len_r4', 'index_r4'),

            DataPaddingProcess('image_og', target_size),
            DataPaddingProcess('image_r1', target_size),
            DataPaddingProcess('image_r2', target_size),
            DataPaddingProcess('image_r3', target_size),
            #DataPaddingProcess('image_r4', target_size),

            ImageSaveProcess('image_og', 'image_save_path_og'),
            ImageSaveProcess('image_r1', 'image_save_path_r1'),
            ImageSaveProcess('image_r2', 'image_save_path_r2'),
            ImageSaveProcess('image_r3', 'image_save_path_r3'),
            #ImageSaveProcess('image_r4', 'image_save_path_r4'),
            LabelSaveProcess('label_og', 'label_save_path_og'),
            LabelSaveProcess('label_r1', 'label_save_path_r1'),
            LabelSaveProcess('label_r2', 'label_save_path_r2'),
            LabelSaveProcess('label_r3', 'label_save_path_r3'),
            #LabelSaveProcess('label_r4', 'label_save_path_r4'),
            IndexSaveProcess('index_og', 'index_save_path'),
            IndexSaveProcess('index_r1', 'index_save_path'),
            IndexSaveProcess('index_r2', 'index_save_path'),
            IndexSaveProcess('index_r3', 'index_save_path'),
            #IndexSaveProcess('index_r4', 'index_save_path')
        ])
        
    def __call__(self, params):
        
        json_path = params['in_json']
        with open(json_path, 'rb') as json_file:
            json_arr = json.load(json_file)

        for json_obj in json_arr:

            result = {}
            result['title'] = json_obj['title']
            result['background_path'] = params['in_dir'] + json_obj['background_path']
            result['label_og'] = np.array([json_obj['faces'][person]['emote']['bbox'] for person in json_obj['faces']])
            result['label_len_og'] = result['label_og'].shape[0]

            base_name = params['prefix'] + '_' + json_obj['title']
        
            result['image_save_path_og'] = params['out_image_dir'] + base_name + '.png'
            result['image_save_path_r1'] = params['out_image_dir'] + base_name + '_crop1.png'
            result['image_save_path_r2'] = params['out_image_dir'] + base_name + '_crop2.png'
            result['image_save_path_r3'] = params['out_image_dir'] + base_name + '_crop3.png'
            result['image_save_path_r4'] = params['out_image_dir'] + base_name + '_crop4.png'
        
            result['label_save_path_og'] = params['out_label_dir'] + base_name + '.txt'
            result['label_save_path_r1'] = params['out_label_dir'] + base_name + '_crop1.txt'
            result['label_save_path_r2'] = params['out_label_dir'] + base_name + '_crop2.txt'
            result['label_save_path_r3'] = params['out_label_dir'] + base_name + '_crop3.txt'
            result['label_save_path_r4'] = params['out_label_dir'] + base_name + '_crop4.txt'

            result['index_og'] = base_name
            result['index_r1'] = base_name + '_crop1'
            result['index_r2'] = base_name + '_crop2'
            result['index_r3'] = base_name + '_crop3'
            result['index_r4'] = base_name + '_crop4'

            result['index_save_path'] = params['out_index']
        
            self.sequence(result)
        
    
        
class LabeledDataset(Dataset):
    def __init__(self, indexpath, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        with open(indexpath, 'r') as indexfile:
            self.lines = indexfile.readlines()
            self.lines = [line[:-1] for line in self.lines]
            
        self.sequence = ProcessSequence([
            ImageLoadProcess('background_path', 'image'),
            LabelLoadProcess('label_path', 'label', 'label_len'),
            DataToTensorProcess('image', 'label', 'label_len')
        ])
            
        
    def __getitem__(self, index):
        result = {}
        result['title'] = self.lines[index]
        result['background_path'] = self.image_dir + self.lines[index] + '.png'
        result['label_path'] = self.label_dir + self.lines[index] + '.txt'
        
        self.sequence(result)
        
        return result

    def __len__(self):
        return len(self.lines)
    
    def shuffle(self):
        random.shuffle(self.lines)
        
#class ImageDataset(Dataset):
#    pass

class VideoDataset(Dataset):
    def __init__(self, image_dir, from_size, to_size, splits = (1, 1)):
        self.image_dir = image_dir
        
        self.splits = splits
        self.splits_count = splits[0] * splits[1]
        self.split_unit = (int(from_size[0] / (splits[0] + 1)), int(from_size[1] / (splits[1] + 1)))
        self.crop_size = [self.split_unit[0] * 2, self.split_unit[1] * 2]
        
        image_list = listdir(image_dir)
        self.image_list = []
        for image_name in image_list:
            if fnmatch(image_name, '*.png') is True:
                self.image_list.append(image_name)
        
        self.sequence = ProcessSequence([
            ImageLoadProcess('background_path', 'image_og'),

            KeyCopyProcess('image_og', 'image'),
            DataCropProcess('image', key_bbox = 'crop_bbox'),
            DataResizeProcess('image', key_label = None, from_size = self.crop_size, to_size = to_size),
            DataPaddingProcess('image', to_size),

            DataToTensorProcess('image', key_label = None, key_label_len = None)
        ])
    
    def __getitem__(self, index):
        
        image_index = index // self.splits_count
        split_index = index % self.splits_count
        split_x_index = split_index % self.splits[0]
        split_y_index = split_index // self.splits[0]
        
        split_x1 = self.split_unit[0] * split_x_index
        split_x2 = split_x1 + self.crop_size[0]
        split_y1 = self.split_unit[1] * split_y_index
        split_y2 = split_y1 + self.crop_size[1]
        
        result = {}
        result['title'] = self.image_list[image_index]
        result['background_path'] = self.image_dir + self.image_list[image_index]
        result['crop_bbox'] = np.array([split_x1, split_y1, split_x2, split_y2])
        result['crop_size'] = np.array(self.crop_size)
        result['split_index'] = split_index
        
        self.sequence(result)
        
        result['image_og'] = np.array(result['image_og'])
        
        return result
        
    def __len__(self):
        return len(self.image_list) * self.splits_count
        

class URLDataset(Dataset):
    pass



