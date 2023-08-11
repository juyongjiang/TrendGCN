import os
import sys
import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime

from model.generator import DAAGCN as Generator
from model.discriminator import Discriminator, Discriminator_RF
from trainer import Trainer

from dataloader import get_dataloader, get_dataloader_meta_la
from utils.metrics import MAE_torch
from utils.util import *
from utils.adj_dis_matrix import get_adj_dis_matrix, norm_Adj as norm_adj

#*************************************************************************#
Mode = 'Train'
DEBUG = 'True'
DATASET = 'PEMS04'      # PEMS03 or PEMS04 or PEMS07 or PEMS08 or METR-LA or PEMS-Bay
MODEL = 'TrendGCN'
ADJ_MATRIX = './dataset/{}/{}.csv'.format(DATASET, DATASET)
#*************************************************************************#

# get configuration
config_file = './config/{}.conf'.format(DATASET)
print('Reading configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

def get_arguments():
    # parser
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--mode', default=Mode, type=str)
    parser.add_argument('--debug', default=DEBUG, type=eval)
    parser.add_argument('--model', default=MODEL, type=str)
    parser.add_argument('--adj_file', default=ADJ_MATRIX, type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    # data
    parser.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    parser.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    parser.add_argument('--lag', default=config['data']['lag'], type=int)
    parser.add_argument('--horizon', default=config['data']['horizon'], type=int)
    parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    parser.add_argument('--tod', default=config['data']['tod'], type=eval)
    parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    parser.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
    # model
    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    # train
    parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    parser.add_argument('--seed', default=config['train']['seed'], type=int)
    parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    parser.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
    # test
    parser.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    parser.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
    # log
    parser.add_argument('--log_dir', default='./', type=str)
    parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
    parser.add_argument('--plot', default=config['log']['plot'], type=eval)
    args = parser.parse_args()
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
    
    return args

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)
    
    return model

if __name__ == "__main__":
    args = get_arguments()
    args.device = get_device(args)
    # init generator and discriminator model
    generator = Generator(args)
    generator = generator.to(args.device)
    generator = init_model(generator)

    discriminator = Discriminator(args)
    discriminator = discriminator.to(args.device)
    discriminator = init_model(discriminator)

    discriminator_rf = Discriminator_RF(args)
    discriminator_rf = discriminator_rf.to(args.device)
    discriminator_rf = init_model(discriminator_rf)

    if args.dataset in ['METR-LA', 'PEMS-Bay']:
        train_loader, val_loader, test_loader, scaler = get_dataloader_meta_la(args,
                                                                    normalizer=args.normalizer,
                                                                    tod=args.tod,
                                                                    dow=False,
                                                                    weather=False,
                                                                    single=False)   
    # load dataset X = [B', W, N, D], Y = [B', H, N, D]
    else:
        train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                                    normalizer=args.normalizer,
                                                                    tod=args.tod, 
                                                                    dow=False,
                                                                    weather=False, 
                                                                    single=False)
    # get norm adj_matrix, norm dis_matrix
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if args.dataset.lower() == 'pems03':
        adj_matrix, dis_matrix = get_adj_dis_matrix(args.adj_file, args.num_nodes, False, "./dataset/PEMS03/PEMS03.txt")
        norm_adj_matrix, norm_dis_matrix = TensorFloat(norm_adj(adj_matrix)), TensorFloat(norm_adj(dis_matrix))
    elif args.dataset.lower() in ['metr-la', 'pems-bay']:
        norm_adj_matrix, norm_dis_matrix = None, None
    else:
        adj_matrix, dis_matrix = get_adj_dis_matrix(args.adj_file, args.num_nodes, False)
        norm_adj_matrix, norm_dis_matrix = TensorFloat(norm_adj(adj_matrix)), TensorFloat(norm_adj(dis_matrix))

    # loss function
    if args.loss_func == 'mask_mae':
        loss_G = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss_G = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss_G = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError

    loss_D = torch.nn.BCELoss()

    # optimizer
    optimizer_G = torch.optim.Adam(params=generator.parameters(), 
                                   lr=args.lr_init, 
                                   eps=1.0e-8, 
                                   weight_decay=0, 
                                   amsgrad=False)

    optimizer_D = torch.optim.Adam(params=discriminator.parameters(), 
                                   lr=args.lr_init*0.1,
                                   eps=1.0e-8, 
                                   weight_decay=0, 
                                   amsgrad=False)

    optimizer_D_RF = torch.optim.Adam(params=discriminator_rf.parameters(), 
                                   lr=args.lr_init*0.1,
                                   eps=1.0e-8, 
                                   weight_decay=0, 
                                   amsgrad=False)
    
    # learning rate decay scheduler
    lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF = None, None, None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_G,
                                                              milestones=lr_decay_steps,
                                                              gamma=args.lr_decay_rate)

        lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D,
                                                              milestones=lr_decay_steps,
                                                              gamma=args.lr_decay_rate)

        lr_scheduler_D_RF = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D_RF,
                                                                 milestones=lr_decay_steps,
                                                                 gamma=args.lr_decay_rate)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

    # config log path
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_dir, 'log', args.dataset, current_time)
    args.log_dir = log_dir

    # model training or testing
    trainer = Trainer(args,
                      generator, discriminator, discriminator_rf, 
                      train_loader, val_loader, test_loader, scaler,
                      norm_dis_matrix, 
                      loss_G, loss_D, 
                      optimizer_G, optimizer_D, optimizer_D_RF, 
                      lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF)
    
    if args.mode.lower() == 'train':
        trainer.train()
    elif args.mode.lower() == 'test':
        # generator.load_state_dict(torch.load('./log/{}/20221128054144/best_model.pth'.format(args.dataset)))
        print("Load saved model")
        trainer.test(generator, norm_dis_matrix, trainer.args, test_loader, scaler, trainer.logger, path=f'./log/{args.dataset}/20221130052054/')
    else:
        raise ValueError
