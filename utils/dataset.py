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
    def __call__(self, params, meta):
        if meta is None:
            return None

        output = {}

        output['title'] = meta['title']
        output['background_path'] = meta['background_path']
        output['faces'] = [meta['faces'][person]['emote']['bbox'] for person in meta['faces']]

        return output

class ResizingProcess(object):
    def __init__(self, size, opt = PIL.Image.NEAREST):
        self.size = size
        self.opt = opt

    def __call__(self, params, meta):
        if meta is None:
            return None

        try :
            meta['background'] = PIL.Image.open(params['in_dir'] + meta['background_path'])
            width, height = meta['background'].size
            width_rate = self.size[0] / float(width)
            height_rate = self.size[1] / float(height)

            meta['faces'] = [[bbox[0] * width_rate, bbox[1] * height_rate, 
                bbox[2] * width_rate, bbox[3] * height_rate] for bbox in meta['faces']]
                
            meta['background'] = meta['background'].resize(self.size, self.opt)
            return meta
        except IOError:
            print('cannot resize image :', params['in_dir'] + meta['background_path'])
            return None

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
            output['image'] = output['image'].permute(2, 0, 1)
        except:
            return None

        return output

    def __len__(self):
        return len(self.lines)
    
    def shuffle(self):
        random.shuffle(self.lines)
        






