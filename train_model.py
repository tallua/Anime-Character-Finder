
# %%
## Import
import torch
from torch import autograd
from torch.utils.data import DataLoader

import argparse
import sys
import json
import gc
import numpy as np
import datetime
from collections import Counter

from utils.dataset import LabeledDataset
from utils.model import YoloV3, YoloLoss
from utils.postprocess import PostProcessor

# %%
## Define Training Routine
def train(model, loss_func, optimizer, scheduler, train_context, train_config, epochs):
    
    postProcessor = PostProcessor()
    
    if train_config['log']['tb_enable']:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir = train_config['log']['tb_dir'])
    else:
        tb_writer = None
    
    for _ in range(0, epochs):
        # training step
        model.train()
        torch.autograd.set_detect_anomaly(train_config['enable_anomaly_detection'])
        
        if train_config['log']['console_enable']:
            print('epoch : ', train_context['epoch'])
            print('    time : ', datetime.datetime.now().time())
            print('    lr : ', train_context['lr'])
        if tb_writer is not None:
            tb_writer.add_scalar('Step/Learning Rate', train_context['lr'], train_context['epoch'])
        
        losses = []
        obj_losses = []
        coord_losses = []
        for idx, batches in enumerate(train_context['train_loader']):
            image = batches['image'].to(train_context['device'], dtype = train_context['dtype'])
            labels = batches['label'].to(train_context['device'], dtype = train_context['dtype'])
            label_len = batches['label_len'].to(train_context['device'], dtype = torch.long)
            
            # forward
            out1, out2, out3 = model(image)
       
            # clear optimizer
            optimizer.zero_grad()
        
            # loss
            loss, obj_loss, coord_loss = loss_func(torch.cat((out1, out2, out3), 1), labels, label_len)
            losses.append(loss.item())
            obj_losses.append(obj_loss.item())
            coord_losses.append(coord_loss.item())
            
            # backward
            loss.backward()
            optimizer.step()
            
            # cleanup
            del image, labels, label_len
            del out1, out2, out3
            del loss, obj_loss, coord_loss
            gc.collect()
            torch.cuda.empty_cache()
    
        # print loss
        train_len = train_context['train_set'].__len__()
        avg_loss = np.sum(losses) / train_len if len(losses) is not 0 else 0
        avg_obj_loss = np.sum(obj_losses) / train_len if len(obj_losses) is not 0 else 0
        avg_coord_loss = np.sum(coord_losses) / train_len if len(coord_losses) is not 0 else 0
        
        train_loss = avg_loss
        
        
        if train_config['log']['console_enable']:
            print('    t_loss : ', avg_loss)
            print('    t_obj_loss : ', avg_obj_loss)
            print('    t_coord_loss : ', avg_coord_loss)
        if tb_writer is not None:
            tb_writer.add_scalar('Loss/Training Loss', avg_loss, train_context['epoch'])
            tb_writer.add_scalar('Loss/Training Object Loss', avg_obj_loss, train_context['epoch'])
            tb_writer.add_scalar('Loss/Training Coord Loss', avg_coord_loss, train_context['epoch'])
        

        # validate step
        with torch.no_grad():
            model.eval()
            torch.autograd.set_detect_anomaly(False)
            
            if train_config['validation']['target']['start_epoch'] <= train_context['epoch']:
                enable_accuracy_test = True
            else:
                enable_accuracy_test = False
                
            losses = []
            obj_losses = []
            coord_losses = []
            if enable_accuracy_test:
                accs = Counter({})
                
            for idx, batches in enumerate(train_context['valid_loader']):
                image = batches['image'].to(train_context['device'], dtype = train_context['dtype'])
                labels = batches['label'].to(train_context['device'], dtype = train_context['dtype'])
                label_len = batches['label_len'].to(train_context['device'], dtype = torch.long)
            
                out1, out2, out3 = model(image)
                pred = torch.cat((out1, out2, out3), 1)
        
                loss, obj_loss, coord_loss = loss_func(pred, labels, label_len)
                losses.append(loss.item())
                obj_losses.append(obj_loss.item())
                coord_losses.append(coord_loss.item())
                
                if enable_accuracy_test:
                    prediction = {}
                    prediction['pred'] = pred.cpu().detach().squeeze(0).numpy()
                    prediction['label'] = batches['label'].cpu().squeeze(0).numpy()
                    prediction['label_len'] = batches['label_len'].cpu().squeeze(0).numpy()
                    
                    post_config = train_config['validation']['post']
                
                    bboxes = postProcessor.CUSTOM1(prediction['pred'], post_config)
                    acc = postProcessor.calcAccuracyMap(prediction['label'], prediction['label_len'], bboxes, post_config)
                    accs = accs + Counter(acc)
            
                # cleanup
                del image, labels, label_len
                del out1, out2, out3
                del loss, obj_loss, coord_loss
                if enable_accuracy_test:
                    del prediction, bboxes
                gc.collect()
                torch.cuda.empty_cache()
    
            # print validation loss
            valid_len = train_context['valid_set'].__len__()
            avg_loss = np.sum(losses) / valid_len if len(losses) is not 0 else 0
            avg_obj_loss = np.sum(obj_losses) / valid_len if len(obj_losses) is not 0 else 0
            avg_coord_loss = np.sum(coord_losses) / valid_len if len(coord_losses) is not 0 else 0
                
            if train_config['log']['console_enable']:
                print('    v_loss : ', avg_loss)
                print('    v_obj_loss  : ', avg_obj_loss)
                print('    v_coord_loss  : ', avg_coord_loss)
                
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/Validation Loss', avg_loss, train_context['epoch'])
                tb_writer.add_scalar('Loss/Validation Object Loss', avg_obj_loss, train_context['epoch'])
                tb_writer.add_scalar('Loss/Validation Coord Loss', avg_coord_loss, train_context['epoch'])
                    
            if enable_accuracy_test:
                tp = accs['true positive']
                fn = accs['false negative']
                fp = accs['false positive'] + accs['duplicate']
                accuracy = tp / (tp + fn + fp)
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                
                # print accuracy
                if train_config['log']['console_enable']:
                    print('    accs : ', accs)
                    print('    accuracy : ', accuracy)
                    print('    recall : ', recall)
                    print('    precision : ', precision)
                if tb_writer is not None:
                    tb_writer.add_scalar('Accuracy/Accuracy', accuracy, train_context['epoch'])
                    tb_writer.add_scalar('Accuracy/Recall', recall, train_context['epoch'])
                    tb_writer.add_scalar('Accuracy/Precision', precision, train_context['epoch'])
                        
                # save model if matching target
                if (accuracy >= train_config['validation']['target']['accuracy'] and 
                    recall >= train_config['validation']['target']['recall'] and 
                    precision >= train_config['validation']['target']['precision']):
                    
                    output_dir = train_config['validation']['target']['save_dir']
                    model_name = train_config['validation']['target']['model_prefix'] + str(train_context['epoch'])
                    torch.save(model, output_dir + model_name + '.dat')
                            
        # save model
        chkpoint_start = train_config['checkpoint']['start_epoch']
        chkpoint_interval = train_config['checkpoint']['interval_epoch']
        if (train_context['epoch'] >= chkpoint_start and 
            (train_context['epoch'] - chkpoint_start) % chkpoint_interval is 0):
            
            train_context['last_checkpoint'] = train_config['checkpoint']
                    
            output_dir = train_config['checkpoint']['save_dir']
            model_name = train_config['checkpoint']['model_prefix'] + str(train_context['epoch'])
            torch.save(model, output_dir + model_name + '.dat')
            
        # update context
        if train_context['epoch'] >= train_config['plan']['lr_decay_start_epoch']:
            train_context['lr_window'].append(train_loss)
            
            if len(train_context['lr_window']) > train_config['plan']['lr_window_size']:
                train_context['lr_window'] = train_context['lr_window'][- train_config['plan']['lr_window_size']:]
            
                if np.mean(train_context['lr_window']) * train_config['plan']['lr_threshold'] < train_loss:
                    train_context['lr'] = train_context['lr'] * train_config['plan']['lr_decay_rate']
                    train_context['lr_window'] = []
                
#            train_context['lr'] = train_context['lr'] * train_config['plan']['lr_decay_rate']
            
        scheduler.step()
        
        train_context['epoch'] += 1
        
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
# %%
def create_config():

    parser = argparse.ArgumentParser(description = 'Train model')

    parser.add_argument('-config', help='path to config file', action='store', default='./config/config.json',  type=str)
    parser.add_argument('-index', help='index of training set', type=str)
    parser.add_argument('-epoch', help='total epoch on training', type=int)
    parser.add_argument('-lr_init', help='initial learning rate on training', type=float)
    parser.add_argument('-lr_decay', help='ly decay on training', type=float)
    parser.add_argument('-out_checkpoint', help='output directory for checkpoint', type=str)
    parser.add_argument('-out_target', help='output directory for target', type=str)

    parsed = parser.parse_args()

    with open(parsed.config, "r") as config_file:
        main_config = json.load(config_file)
    
    if parsed.index is not None:
        main_config['train']['set']['index'] = parsed.index
    if parsed.epoch is not None:
        main_config['train']['plan']['epochs'] = parsed.epoch
    if parsed.lr_init is not None:
        main_config['train']['plan']['lr_init'] = parsed.lr_init
    if parsed.lr_decay is not None:
        main_config['train']['plan']['lr_decay_rate'] = parsed.lr_decay
    if parsed.out_checkpoint is not None:
        main_config['train']['checkpoint']['save_dir'] = parsed.out_checkpoint
    if parsed.out_target is not None:
        main_config['train']['validation']['target']['save_dir'] = parsed.out_target

    return main_config


# %%
if __name__ == '__main__':

    ## Load Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float

    main_config = create_config()

    try:
        model_config = main_config['model']
        train_config = main_config['train']
        valid_config = main_config['train']['validation']
        loss_config = main_config['train']['loss']
    except NameError:
        assert False, ('Failed to load config file')
    except KeyError:
        assert False, ('Failed to find key on config file')

    model_config['device'] = device
    model_config['dtype'] = dtype
    model_config['attrib_count'] = 5 + model_config['class_count']

    loss_config['device'] = device
    loss_config['dtype'] = dtype
    loss_config['attrib_count'] = model_config['attrib_count']

    ## Create Context
    train_context = { }

    train_context['device'] = device
    train_context['dtype'] = dtype

    train_context['train_set'] = LabeledDataset(train_config['set']['index'], 
                                              train_config['set']['image_dir'], 
                                              train_config['set']['label_dir'])
    train_context['train_loader'] = DataLoader(train_context['train_set'], 
                                               batch_size = train_config['set']['batch_size'], 
                                               num_workers = train_config['set']['num_workers'],
                                               shuffle = True)

    train_context['valid_set'] = LabeledDataset(valid_config['set']['index'], 
                                              valid_config['set']['image_dir'], 
                                              valid_config['set']['label_dir'])
    train_context['valid_loader'] = DataLoader(train_context['valid_set'], 
                                               batch_size = valid_config['set']['batch_size'], 
                                               num_workers = valid_config['set']['num_workers'],
                                               shuffle = False)

    train_context['epoch'] = 0
    train_context['last_checkpoint'] = 0
    train_context['lr'] = train_config['plan']['lr_init']
    train_context['lr_window'] = []

    ## Build Model & Components
    model = YoloV3(model_config)
    model = model.to(model_config['device'])

    loss_func = YoloLoss(loss_config)

    lr_func = lambda epoch: train_context['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr = train_context['lr'])
    #train_context['optimizer'] = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_func, last_epoch = -1)

    ## Run
    train(model, loss_func, optimizer, scheduler, train_context, train_config, train_config['plan']['epochs'])
