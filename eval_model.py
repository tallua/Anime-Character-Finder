
# %%
## Import

import torch
from torch import autograd
from torch.utils.data import DataLoader

import argparse
import json
import gc
import numpy as np

from utils.dataset import LabeledDataset, VideoDataset
from utils.model import YoloV3, YoloLoss
from utils.postprocess import PostProcessor

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

import os
import fnmatch
from collections import Counter

import urllib
from io import BytesIO

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

from random import randint

import datetime


# %%
## Define Eval Routines



def eval_images():
    ## Create Context

    
    ## Load Model & Components

    pass
    


def eval_video():
    ## Create Context

    
    ## Load Model & Components

    pass


def eval_label():
    ## Create Context

    
    ## Load Model & Components

    pass





# %%
## Config
def create_config():

    parser = argparse.ArgumentParser(description = 'Eval model')

    parser.add_argument('-config', help='path to config file. default:"./config/config.json"', action='store', default='./config/config.json',  type=str)
    parser.add_argument('-mode', help='output directory', action='store', default='./config/config.json',  type=str)
    parser.add_argument('-output', help='output directory', action='store', default='./config/config.json',  type=str)
    parser.add_argument('-timestamp', help='output directory', action='store', default='', nargs="+", type=str)

    parser.add_argument('-plot_opts', help='output directory', action='store', default='', nargs="+", type=str)

    parser.add_argument('-font_size', help='output directory', action='store', default='./config/config.json',  type=int)
    parser.add_argument('-border_size', help='output directory', action='store', default='./config/config.json',  type=int)

    parsed = parser.parse_args()

    with open(parsed.config, "r") as config_file:
        main_config = json.load(config_file)
    


    

    return main_config


# %%
## Main

if __name__ == '__main__':

    ## Load Config    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float

    main_config = create_config()



    

