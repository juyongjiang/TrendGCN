import torch
import torch.nn as nn
import math
import os
import time
import copy
import numpy as np
from utils.util import get_logger
from utils.metrics import All_Metrics

class Trainer(object):
    def __init__(self, 
                 args,
                 generator, discriminator, discriminator_rf,
                 train_loader, val_loader, test_loader, scaler,
                 norm_dis_matrix,
                 loss_G, loss_D, 
                 optimizer_G, optimizer_D, optimizer_D_RF, 
                 lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF):

        super(Trainer, self).__init__()
        self.args = args
        self.num_nodes = args.num_nodes

        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_rf = discriminator_rf
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_D_RF = optimizer_D_RF
        self.lr_scheduler_G = lr_scheduler_G
        self.lr_scheduler_D = lr_scheduler_D
        self.lr_scheduler_D_RF = lr_scheduler_D_RF


        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.norm_dis_matrix = norm_dis_matrix

        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)

        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png') # when plot=True
        
        # log info
        if os.path.isdir(args.log_dir) == False:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info(f"Argument: {args}")
        for arg, value in sorted(vars(args).items()):
            self.logger.info(f"{arg}: {value}")

    def train_epoch(self, epoch):
        self.generator.train()
        total_loss_G = 0
        total_loss_D = 0
        total_loss_D_RF = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_size = data.shape[0]
            data = data[..., :self.args.input_dim] # [B'', W, N, 1]
            label = target[..., :self.args.output_dim]  # # [B'', H, N, 1]

            # Adversarial ground truths
            cuda = True if torch.cuda.is_available() else False
            TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            valid = torch.autograd.Variable(TensorFloat(batch_size*(self.args.lag + self.args.horizon), 1).fill_(1.0), requires_grad=False)
            fake = torch.autograd.Variable(TensorFloat(batch_size*(self.args.lag + self.args.horizon), 1).fill_(0.0), requires_grad=False)

            valid_rf = torch.autograd.Variable(TensorFloat(batch_size*self.args.num_nodes, 1).fill_(1.0), requires_grad=False)
            fake_rf = torch.autograd.Variable(TensorFloat(batch_size*self.args.num_nodes, 1).fill_(0.0), requires_grad=False)

            #-------------------------------------------------------------------
            # Train Generator 
            #-------------------------------------------------------------------
            self.optimizer_G.zero_grad()
                        
            # data and target shape: B, W, N, F, and B, H, N, F; output shape: B, H, N, F (F=1)
            output = self.generator(data, self.norm_dis_matrix)
            if self.args.real_value: # it is depended on the output of model. If output is real data, the label should be reversed to real data
                label = self.scaler.inverse_transform(label)
            
            fake_input = torch.cat((data, self.scaler.transform(output)), dim=1) # [B'', W, N, 1] // [B'', H, N, 1] -> [B'', W+H, N, 1]
            true_input = torch.cat((data, self.scaler.transform(label)), dim=1) if self.args.real_value else torch.cat((data, label), dim=1)

            fake_input_rf = self.scaler.transform(output) # [B'', W, N, 1] // [B'', H, N, 1] -> [B'', W+H, N, 1]
            true_input_rf = self.scaler.transform(label) if self.args.real_value else label

            # fake_input_rf_real = torch.cat((self.scaler.inverse_transform(data), output), dim=1) # [B'', W+H, N, 1]
            # true_input_rf_real = torch.cat((self.scaler.inverse_transform(data), label), dim=1)
            # print(true_input_rf[0][0][:10][:10])
            # input('check')
            # print(fake_input_rf.shape)
            # input('check')
            # print(self.scaler.transform(output)[0, 0].squeeze(), data[0, 0].squeeze()) # , label[0, 0].squeeze())
            # input('check')
            # print(self.loss_G(output.cuda(), label).item(), self.loss_D(self.discriminator(fake_input), valid).item())
            # input('check')
            # print(self.discriminator(fake_input).squeeze()[:64])
            # input('check')
            # self.loss_D(self.discriminator(fake_input), valid) 
            # 
            
            loss_G = self.loss_G(output.cuda(), label) + 0.01 * self.loss_D(self.discriminator(fake_input), valid) + self.loss_D(self.discriminator_rf(fake_input_rf), valid_rf)
            loss_G.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.max_grad_norm)
            self.optimizer_G.step()
            total_loss_G += loss_G.item()

            #-------------------------------------------------------------------
            # Train Discriminator 
            #-------------------------------------------------------------------
            self.optimizer_D.zero_grad()
            real_loss = self.loss_D(self.discriminator(true_input), valid)
            fake_loss = self.loss_D(self.discriminator(fake_input.detach()), fake)
            # print(self.discriminator(true_input).squeeze()[:64])
            # print(self.discriminator(fake_input).squeeze()[:64])
            # input('check')
            loss_D = 0.5 * (real_loss + fake_loss)
            loss_D.backward()
            self.optimizer_D.step() 
            total_loss_D += loss_D.item()

            #-------------------------------------------------------------------
            # Train Discriminator_RF
            #-------------------------------------------------------------------
            self.optimizer_D_RF.zero_grad()
            real_loss_rf = self.loss_D(self.discriminator_rf(true_input_rf), valid_rf)
            fake_loss_rf = self.loss_D(self.discriminator_rf(fake_input_rf.detach()), fake_rf)
            # print(self.discriminator_rf(true_input_rf).squeeze()[:64])
            # print(self.discriminator_rf(fake_input_rf).squeeze()[:64])
            # input('check')
            loss_D_RF = 0.5 * (real_loss_rf + fake_loss_rf)
            loss_D_RF.backward()
            self.optimizer_D_RF.step() 
            total_loss_D_RF += loss_D_RF.item()

            # log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Generator Loss: {:.6f} Pred Discriminator Loss: {:.6f} RelFlow Discriminator Loss: {:.6f}'.format(
                                 epoch, 
                                 batch_idx, self.train_per_epoch, 
                                 loss_G.item(), loss_D.item(), loss_D_RF.item()))
        
        train_epoch_loss_G = total_loss_G / self.train_per_epoch # average generator loss
        train_epoch_loss_D = total_loss_D / self.train_per_epoch # average discriminator loss
        train_epoch_loss_D_RF = total_loss_D_RF / self.train_per_epoch # average discriminator loss

        self.logger.info('**********Train Epoch {}: Averaged Generator Loss: {:.6f}, Averaged Pred Discriminator Loss: {:.6f}, Averaged RelFlow Discriminator Loss: {:.6f}'.format(
                         epoch, 
                         train_epoch_loss_G,
                         train_epoch_loss_D,
                         train_epoch_loss_D_RF))

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()
            self.lr_scheduler_D_RF.step()
        
        return train_epoch_loss_G, train_epoch_loss_D, train_epoch_loss_D_RF  

    def val_epoch(self, epoch, val_dataloader):
        self.generator.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim] # [B'', W, N, 1]
                label = target[..., :self.args.output_dim] # [B'', H, N, 1]
                output = self.generator(data, self.norm_dis_matrix)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss_G(output.cuda(), label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        
        return val_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list_G = []
        train_loss_list_D = []
        train_loss_list_D_RF = []
        val_loss_list = []

        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss_G, train_epoch_loss_D, train_epoch_loss_D_RF = self.train_epoch(epoch)
            
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            train_loss_list_G.append(train_epoch_loss_G)
            train_loss_list_D.append(train_epoch_loss_D)
            train_loss_list_D_RF.append(train_epoch_loss_D_RF)
            val_loss_list.append(val_epoch_loss)

            if train_epoch_loss_G > 1e6 or train_epoch_loss_D > 1e6 or train_epoch_loss_D_RF > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
                
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs! Training stops!".format(self.args.early_stop_patience))
                    # break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.generator.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        # if not self.args.debug:
        self.save_checkpoint()

        # test
        self.generator.load_state_dict(best_model)
        # self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.generator, self.norm_dis_matrix, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.generator.state_dict(),
            'optimizer': self.optimizer_G.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, norm_dis_matrix, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(os.path.join(path, 'best_model.pth')) # path = args.log_dir
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim] # [B'', W, N, 1]
                label = target[..., :args.output_dim] # [B'', H, N, 1]
                output = model(data, norm_dis_matrix)
                y_true.append(label) # [B'', H, N, 1]
                y_pred.append(output) # [B'', H, N, 1]
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        
        # save predicted results as numpy format
        np.save(os.path.join(args.log_dir, '{}_true.npy'.format(args.dataset)), y_true.cpu().numpy())
        np.save(os.path.join(args.log_dir, '{}_pred.npy'.format(args.dataset)), y_pred.cpu().numpy())

        # each horizon point
        for t in range(y_true.shape[1]): # H
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape*100))
        # average all horizon point
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape*100))