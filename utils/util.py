import os
import torch
import random
import numpy as np
import logging
from datetime import datetime
from utils.metrics import MAE_torch

def get_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda", args.gpu_id)
    else:
        device = torch.device("cpu")
    set_seed(args) # reproductibility

    return device

def set_seed(args):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)   
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')


def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))
    
    return allocated_memory, cached_memory

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss

def get_logger(root, name=None, debug=True):
    # when debug is true, show DEBUG and INFO in screen
    # when debug is false, show DEBUG in file and info in both screen&file
    # INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    # critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
        
    return logger