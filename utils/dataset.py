from os import mkdir
from pathlib import Path

import json

import torch
from torch.utils.data import Dataset

import copy

import PIL.Image
import numpy as np

import random


class DetectionPreProcessor:
    def __init__(self):
        self.procs = []

    def addProcessor(self, process):
        self.procs.append(process)

    def process(self, params = None):
        jsonpath = params['in_json']
        with open(jsonpath, 'rb') as jsonfile:
            json_obj = json.load(jsonfile)

        for obj in json_obj:
            for proc in self.procs:
                obj = proc(params, obj)

class PickingProcess(object):
    def __init__(self, postfix = ''):
        self.postfix = postfix
    
    def __call__(self, params, meta):
        if meta is None:
            return None

        output = {}

        output['title'] = meta['title'] + self.postfix
        output['background_path'] = meta['background_path']
        output['faces'] = [meta['faces'][person]['emote']['bbox'] for person in meta['faces']]
        try:
            output['background'] = PIL.Image.open(params['in_dir'] + output['background_path'])
            output['background'] = output['background'].crop(output['background'].getbbox())
        except IOError:
            print('cannot load image :', params['in_dir'] + output['background_path'])
            return None

        return output

class ResizingProcess(object):
    def __init__(self, size, opt = PIL.Image.NEAREST):
        self.size = size
        self.opt = opt

    def __call__(self, params, meta):
        if meta is None:
            return None

        width, height = meta['background'].size
        
        width_rate = self.size[0] / float(width)
        height_rate = self.size[1] / float(height)
        min_rate = min(width_rate, height_rate)
        
        try:
            old_image = meta['background'].resize((int(width * min_rate), int(height * min_rate)), self.opt)
            new_image = PIL.Image.new("RGB", (self.size[0], self.size[1]))
            new_image.paste(old_image, old_image.getbbox())
        except ValueError:
            print(meta['title'])
            print(old_image.getbbox())
        
        meta['faces'] = [[val * min_rate for val in bbox] for bbox in meta['faces']]
        meta['background'] = new_image
        
        return meta

class SavingProcess(object):
    def __call__(self, params, meta):
        if meta is None:
            return None

        try:
            mkdir(params['out_dir'])
        except:
            pass

        filename = params['prefix'] + '_' + meta['title']
        with open(params['out_index'], 'a+') as metafile:
            metafile.write(filename + '\n')

        labelname = params['out_label_dir'] + filename + '.txt'
        with open(labelname, 'w+') as labelfile:
            for bbox in meta['faces']:
                labelfile.write('0')
                for val in bbox:
                    labelfile.write(',%.4f' % val)
                labelfile.write('\n')

        imagename = params['out_image_dir'] + filename + '.png'
        meta['background'].save(imagename, 'PNG')
        
        return meta


class CropResizingProcess(object):
    def __init__(self, size, opt = PIL.Image.NEAREST):
        self.size = size
        self.opt = opt

    def __call__(self, params, meta):
        if meta is None:
            return None

        orig_faces = meta['faces']
        orig_w, orig_h = meta['background'].size
            
        min_w = self.size[0] // 2
        if self.size[0] > orig_w :
            min_w = orig_w
        max_w = orig_w
               
        min_h = self.size[1] // 2
        if self.size[1] > orig_h :
            min_h = orig_h
        max_h = orig_h
                
        if min_w == max_w or min_h == max_h:
            return None
            
        for _ in range(0, 10) :
            new_w = random.randrange(min_w, max_w, 1)
            new_h = random.randrange(min_h, max_h, 1)
            new_x1 = random.randrange(0, max_w - new_w, 1)
            new_x2 = new_x1 + new_w
            new_y1 = random.randrange(0, max_h - new_h, 1)
            new_y2 = new_y1 + new_h
            
            if abs((orig_w / orig_h) - (new_w / new_h)) < 0.5:
                continue
            
            collide = False
            new_faces = []
            for face in orig_faces:
                if (face[0] - new_x1) * (face[2] - new_x1) < 0:
                    collide = True
                    break
                if (face[0] - new_x2) * (face[2] - new_x2) < 0:
                    collide = True
                    break
                if (face[1] - new_y1) * (face[3] - new_y1) < 0:
                    collide = True
                    break
                if (face[1] - new_y2) * (face[3] - new_y2) < 0:
                    collide = True
                    break
                
                new_face = [face[0] - new_x1, face[1] - new_y1, face[2] - new_x1, face[3] - new_y1]
                
                if new_face[0] < 0 or new_face[1] < 0 or new_face[2] < 0 or new_face[3] < 0:
                    continue
                    
                if new_face[0] > new_w or new_face[1] > new_h or new_face[2] > new_w or new_face[3] > new_h:
                    continue
                    
                new_faces.append(new_face)
                    
            if collide is True :
                continue
                    
            if len(new_faces) == 0:
                continue
            
            width_rate = self.size[0] / float(new_w)
            height_rate = self.size[1] / float(new_h)
            min_rate = min(width_rate, height_rate)
            
            try:
                old_image = meta['background'].crop((new_x1, new_y1, new_x2, new_y2))
                old_image = old_image.resize((int(new_w * min_rate), int(new_h * min_rate)), self.opt)
                old_image = old_image.crop(old_image.getbbox())
                new_image = PIL.Image.new("RGB", (self.size[0], self.size[1]))
                new_image.paste(old_image, old_image.getbbox())
            except ValueError:
                print(meta['title'])
                print(old_image.getbbox())
        
        
            meta['faces'] = [[val * min_rate for val in bbox] for bbox in new_faces]
            meta['background'] = new_image
            
            return meta
        return None
        
        
class DetectionFolder(Dataset):
    def __init__(self, indexpath, imagedir, labeldir):
        self.imagedir = imagedir
        self.labeldir = labeldir
        with open(indexpath, 'r') as indexfile:
            self.lines = indexfile.readlines()
            self.lines = [line[:-1] for line in self.lines]

    def __getitem__(self, index):
        output = { }
        output['title'] = self.lines[index]

        imagepath = self.imagedir + output['title'] + '.png'
        labelpath = self.labeldir + output['title'] + '.txt'

        with open(labelpath, 'r') as labelfile:
            labellines = labelfile.readlines()
            output['label'] = np.array([labelline.split(',') for labelline in labellines])
            output['label'] = output['label'].astype(np.float)
            output['label'] = np.array([[(label[1] + label[3]) / 2, (label[2] + label[4]) / 2,
                               label[3] - label[1], label[4] - label[2], label[0]] for label in output['label']])
            output['label'] = torch.from_numpy(output['label'])
            output['label_len'] = torch.tensor(output['label'].shape[0])
            if output['label'].shape[0] > 15:
                output['label'] = output['label'][0:15]
            elif output['label'].shape[0] < 15:
                output['label'] = torch.cat((output['label'].float(), torch.zeros(15 - output['label'].shape[0], 5)), 0)

        try :
            output['image'] = np.array(PIL.Image.open(imagepath)) / 255
            output['image'] = torch.from_numpy(output['image'])
            output['image'] = output['image'].permute(2, 0, 1)[0:3]
        except:
            return None

        return output

    def __len__(self):
        return len(self.lines)
    
    def shuffle(self):
        random.shuffle(self.lines)
        




###
# New ways to process data
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
    
#class DataCropProcess(object):
#    def __init__(self, key_image, key_label = None, key_label_len = None, target_size = None):
#        
#        pass

class RandomDataCropProcess(object):
    def __init__(self, key_image, key_label, key_label_len, key_index, min_size = None, ratio_threshold = 0.3, tries = 10):
        self.key_image = key_image
        self.key_label = key_label
        self.key_label_len = key_label_len
        self.key_index = key_index
        self.min_size = min_size
        self.ratio_threshold = ratio_threshold
        self.tries = tries
        
    def __call__(self, result):
        if (self.key_image not in result) or (self.key_label not in result) or (self.key_label_len not in result):
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
                
        if min_w >= max_w or min_h >= max_h:
            del result[self.key_image] 
            del result[self.key_label] 
            del result[self.key_label_len] 
            del result[self.key_index] 
            
        for _ in range(0, self.tries) :
            new_w = random.randrange(min_w, max_w, 1)
            new_h = random.randrange(min_h, max_h, 1)
            new_x1 = random.randrange(0, max_w - new_w, 1)
            new_x2 = new_x1 + new_w
            new_y1 = random.randrange(0, max_h - new_h, 1)
            new_y2 = new_y1 + new_h

            if abs(1 - (new_w / orig_w)) < self.ratio_threshold:
                continue
            
            collide = False
            new_faces = []
            for face in orig_faces:
                if (face[0] - new_x1) * (face[2] - new_x1) < 0:
                    collide = True
                    break
                if (face[0] - new_x2) * (face[2] - new_x2) < 0:
                    collide = True
                    break
                if (face[1] - new_y1) * (face[3] - new_y1) < 0:
                    collide = True
                    break
                if (face[1] - new_y2) * (face[3] - new_y2) < 0:
                    collide = True
                    break
                
                new_face = [face[0] - new_x1, face[1] - new_y1, face[2] - new_x1, face[3] - new_y1]
                
                if new_face[0] < 0 or new_face[1] < 0 or new_face[2] < 0 or new_face[3] < 0:
                    continue
                    
                if new_face[0] > new_w or new_face[1] > new_h or new_face[2] > new_w or new_face[3] > new_h:
                    continue
                    
                new_faces.append(new_face)
                    
            if collide is True :
                continue
                    
            if len(new_faces) == 0:
                continue
            
            try:
                result[self.key_image] = result[self.key_image].crop((new_x1, new_y1, new_x2, new_y2))
                result[self.key_label][:, 0:4] = [bbox for bbox in new_faces]
                result[self.key_label_len] = len(new_faces)
            except ValueError:
                print('cannot random crop : ', result['title'])
                print('    bbox :', (new_x1, new_y1, new_x2, new_y2))
                del result[self.key_image] 
                del result[self.key_label] 
                del result[self.key_label_len] 
                del result[self.key_index]

            return
        
        del result[self.key_image] 
        del result[self.key_label] 
        del result[self.key_label_len] 
        del result[self.key_index]

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
            KeyCopyProcess('label_og', 'label_r1'),
            KeyCopyProcess('label_og', 'label_r2'),
            KeyCopyProcess('label_len_og', 'label_len_r1'),
            KeyCopyProcess('label_len_og', 'label_len_r2'),

            RandomDataCropProcess('image_r1', 'label_r1', 'label_len_r1', 'index_r1'),
            RandomDataCropProcess('image_r2', 'label_r2', 'label_len_r2', 'index_r2'),

            DataResizeProcess('image_og', 'label_og', keep_ratio = True, to_size = target_size),
            DataResizeProcess('image_r1', 'label_r1', keep_ratio = True, to_size = target_size),
            DataResizeProcess('image_r2', 'label_r2', keep_ratio = True, to_size = target_size),

            DataPaddingProcess('image_og', target_size),
            DataPaddingProcess('image_r1', target_size),
            DataPaddingProcess('image_r2', target_size),

            ImageSaveProcess('image_og', 'image_save_path_og'),
            ImageSaveProcess('image_r1', 'image_save_path_r1'),
            ImageSaveProcess('image_r2', 'image_save_path_r2'),
            LabelSaveProcess('label_og', 'label_save_path_og'),
            LabelSaveProcess('label_r1', 'label_save_path_r1'),
            LabelSaveProcess('label_r2', 'label_save_path_r2'),
            IndexSaveProcess('index_og', 'index_save_path'),
            IndexSaveProcess('index_r1', 'index_save_path'),
            IndexSaveProcess('index_r2', 'index_save_path')
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
        
            result['label_save_path_og'] = params['out_label_dir'] + base_name + '.txt'
            result['label_save_path_r1'] = params['out_label_dir'] + base_name + '_crop1.txt'
            result['label_save_path_r2'] = params['out_label_dir'] + base_name + '_crop2.txt'

            result['index_og'] = base_name
            result['index_r1'] = base_name + '_crop1'
            result['index_r2'] = base_name + '_crop2'

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
    def __init__(self, image_dir, from_size, to_size):

        self.sequence = ProcessSequence([
            ImageLoadProcess('background_path', 'image_og'),

            KeyCopyProcess('image_og', 'image'),
            DataResizeProcess('image', key_label = None, from_size = from_size, to_size = to_size),
            DataPaddingProcess('image', to_size),

            DataToTensorProcess('image', key_label = None, key_label_len = None)
        ])
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class URLDataset(Dataset):
    pass



