from os import mkdir
from pathlib import Path

import json

import torch
from torch.utils.data import Dataset

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
        
        old_image = meta['background'].resize((int(width * min_rate), int(height * min_rate)), self.opt)
        new_image = PIL.Image.new("RGB", (self.size[0], self.size[1]))
        new_image.paste(old_image, old_image.getbbox())
        
        meta['faces'] = [[val * min_rate for val in bbox] for bbox in meta['faces']]
        meta['background'] = new_image

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
            
            old_image = meta['background'].crop((new_x1, new_y1, new_x2, new_y2)).resize((int(new_w * min_rate), int(new_h * min_rate)), self.opt)
            new_image = PIL.Image.new("RGB", (self.size[0], self.size[1]))
            new_image.paste(old_image, old_image.getbbox())
        
        
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
        






