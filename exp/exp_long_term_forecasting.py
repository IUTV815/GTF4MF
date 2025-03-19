from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate,visual_forecast,visual_fea,plot_heatmap
from utils.metrics import metric,MSE,MAE
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
import torch.nn.functional as F

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        print("Model total parameters: {:.2f} M".format(sum(p.numel() for p in model.parameters())/1e+6))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_loader = data_provider(flag,self.args)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss() if (self.args.loss=='MSE' or self.args.loss=='mse') else nn.L1Loss()
        return criterion

    def vali(self, vali_loader,test=None):
        self.model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            total_mse = 0
            if test is not None: total_mae = 0
            total_samples = 0  # To keep track of the total number of samples
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.task == 'MS':
                    batch_x=batch_x
                    batch_y=batch_y[:,:,-1].unsqueeze(-1)
                if self.args.task == 'SS':
                    batch_x=batch_x[:,:,-1].unsqueeze(-1)
                    batch_y=batch_y[:,:,-1].unsqueeze(-1)
                channel=batch_y.shape[-1]
                
                outputs = self.model(batch_x)
                if self.args.task!='MM': outputs=outputs[:,:,-1].unsqueeze(-1)
                
                # Compute MSE and MAE for the current batch
                mse_loss_fn = nn.MSELoss(reduction='sum')  # Sum for accumulating loss
                if test is not None: mae_loss_fn = nn.L1Loss(reduction='sum')  # Sum for accumulating loss
                
                mse = mse_loss_fn(outputs, batch_y)
                if test is not None: mae = mae_loss_fn(outputs, batch_y)
                
                batch_size = batch_y.size(0)  # Get batch size
                total_samples += batch_size  # Accumulate total number of samples
                total_mse += mse.item()  # Add batch MSE to total
                if test is not None: total_mae += mae.item()  # Add batch MAE to total

            # Compute the average loss over all samples (same as mean reduction)
            avg_mse = total_mse / (total_samples*self.args.pred_len*channel)
            avg_mae = total_mae / (total_samples*self.args.pred_len*channel) if test is not None else 0
            
        self.model.train()
        return avg_mse, avg_mae


    def train(self, setting):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.task == 'MS':
                    batch_x=batch_x
                    batch_y=batch_y[:,:,-1].unsqueeze(-1)
                if self.args.task == 'SS':
                    batch_x=batch_x[:,:,-1].unsqueeze(-1)
                    batch_y=batch_y[:,:,-1].unsqueeze(-1)
                
                model_optim.zero_grad(set_to_none=True)
                
                outputs = self.model(batch_x)
                if self.args.task!='MM': outputs=outputs[:,:,-1].unsqueeze(-1)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                    
                loss.backward()
                model_optim.step()
            train_loss = np.average(train_loss)
            vali_mse,vali_mae = self.vali( vali_loader)
            test_mse,test_mae = self.vali( test_loader,test=1)

            print("Epoch: {} Cost time: {:.4f} S | Train Loss: {:.4f} Test MSE: {:.3f} Test MAE: {:.3f}".format(epoch + 1, time.time() - epoch_time, train_loss, test_mse, test_mae))
            early_stopping(vali_mse, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        torch.cuda.empty_cache()
        ## Uncomment below code for save space on device
        # if os.path.exists(os.path.join(os.path.join(path, 'checkpoint.pth'))):
        #     os.remove(os.path.join(os.path.join(path, 'checkpoint.pth')))
        #     print('Model weights deleted.')

    def test(self, setting, test=1):
        test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.task == 'MS':
                    batch_x=batch_x
                    batch_y=batch_y[:,:,-1].unsqueeze(-1)
                if self.args.task == 'SS':
                    batch_x=batch_x[:,:,-1].unsqueeze(-1)
                    batch_y=batch_y[:,:,-1].unsqueeze(-1)
                    
                outputs = self.model(batch_x)
                if self.args.task!='MM': outputs=outputs[:,:,-1].unsqueeze(-1)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
        if len(preds)>0:
            preds=np.concatenate(preds, axis=0)
            trues=np.concatenate(trues, axis=0)
        else:
            preds=preds[0]
            trues=trues[0]
        print('test shape:', preds.shape, trues.shape)
        mae, mse = metric(preds, trues)
        print('mse:{:.3f}, mae:{:.3f}'.format(mse, mae))
        
        # Uncomment what follows to save the test dict: 
        dict_path = f'./test_dict/{self.args.data_type}/{self.args.task}/{self.args.model}/{self.args.seq_len}_{self.args.pred_len}/'
        if not os.path.exists(dict_path):
                os.makedirs(dict_path)
        my_dict = {
            'mse': float(round(mse, 3)),
            'mae': float(round(mae, 3)),
        }
        with open(os.path.join(dict_path, 'records.json'), 'w') as f:
            json.dump(my_dict, f)
        f.close()
        
        return 
        
    def visual_forecasting(self, setting, test=1):
        visual_path = f'./visual/{self.args.data_type}/{self.args.task}/{self.args.model}/{self.args.seq_len}_{self.args.pred_len}/'
        if not os.path.exists(visual_path):
                os.makedirs(visual_path)
        test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model..............\n')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.task == 'MS':
                    batch_x=batch_x
                    batch_y=batch_y[:,:,-1].unsqueeze(-1)
                if self.args.task == 'SS':
                    batch_x=batch_x[:,:,-1].unsqueeze(-1)
                    batch_y=batch_y[:,:,-1].unsqueeze(-1)

                outputs = self.model(batch_x)
                if self.args.task!='MM': outputs=outputs[:,:,-1].unsqueeze(-1)
                step=1
                if i % step == 0:
                    for j in [16,23]:
                        pred = outputs.detach().cpu().numpy()[0, :, j]
                        true = batch_y.detach().cpu().numpy()[0, -self.args.pred_len:, j]
                        input = batch_x.detach().cpu().numpy()[0, :, j]
                        pd = np.concatenate([input, pred])
                        tg= np.concatenate([input, true])
                        # pd = pred
                        # tg= true

                        visual_forecast([pd,'forecast',1],[tg,'target',4], \
                                name=os.path.join(visual_path, str(i) +'_' +str(j)+'_forecast.png'))
        return 