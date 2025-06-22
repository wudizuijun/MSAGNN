import os
from utils.Utils import Utils 
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import torch
from utils.cal_adj_mat import AdjMatCal
from models.AttenGRU import AttGRU
from datetime import datetime
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback

now = datetime.now().strftime("%Y%m%d%H%M%S")

class LitModule(L.LightningModule):
    def __init__(self, model, targets_name=None, targets_idx=None, target_scaler=None,  **kwargs):
        super().__init__()
        self.model = model
        self.args = kwargs
        
        self.target_name = targets_name
        self.target_idx = targets_idx
        self.scaler = target_scaler
        
        self.loss_fn = self._get_loss_fn()
        self.training_step_outputs = []
        self.val_step_outputs = []
        self._loss = {
            'training': [],
            'validate': [],
        }
        # init model
        self._epoch_rec = 0
        self.model.apply(Utils.weight_init)
    
    def format_output(self, pred, real):
        ''' 按照预测的target_idx进行格式化
        Input:
            target: shape (B, N)
            pred: shape (B, N)
        Output:
            pred: shape (B, len(self.target_idx))
            real: shape (B, len(self.target_idx))
        '''
        if self.scaler is not None:
            pred = self.scaler.inverse_transform(pred)
            # real = self.scaler.inverse_transform(real)
        pred = pred[:, self.target_idx]
        real = real[:, self.target_idx]

        return pred, real

    # 对应predict
    def forward(self, batch):
        X, X_d, y = batch
        X = X.permute(0,2,1)
        A_d_theta = atten = A_d = None
        if self.args['model_type'] != 'AttGRU':
            pred = self.model(X, self.adj, X_d) # model input_shape: x(B*seq*N)
        else:
            pred, A_d_theta, A_d, atten, _  = self.model(X, self.adj, X_d) # AttGRU output
        pred, y = self.format_output(pred, y)
        return pred, y, A_d, atten

    def training_step(self, batch, batch_idx):
        """training_step at each batch data
        :param _type_ batch: one batch data, according to the dataloader
        batch: (X, X_d, y), X: (B * N * seq), X_d: (B * N * N * seq) -> abort, y: (B * 1)
        :return _type_: _description_
        """
        X, X_d, y = batch
        X = X.permute(0,2,1)
        A_d = None
        if self.args['model_type'] != 'AttGRU':
            pred = self.model(X, self.adj, X_d) # model input_shape: x(B*seq*N)
        else:
            pred, A_d, _, _, g_loss = self.model(X, self.adj, X_d) # AttGRU output
        pred, y = self.format_output(pred, y)
        if self.args['model_type'] != 'AttGRU':
            loss = self.loss_fn(pred, y, self.adj, A_d)
        else:
            loss = self.loss_fn(pred, y, self.adj, A_d) + torch.mean(g_loss)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation at each batch data
        :param _type_ batch: one batch data, according to the dataloader
        batch: (X, X_d, y), X: (B * N * seq), X_d: (B * N * N * seq) -> abort, y: (B * 1)
        :return _type_: _description_
        """
        X, X_d, y = batch
        X = X.permute(0,2,1)
        A_d = None
        if self.args['model_type'] != 'AttGRU':
            pred = self.model(X, self.adj, X_d) # model input_shape: x(B*seq*N)
        else:
            pred, A_d, _, _, g_loss = self.model(X, self.adj, X_d) # AttGRU output
        pred, y = self.format_output(pred, y)
        if self.args['model_type'] != 'AttGRU':
            loss = self.loss_fn(pred, y, self.adj, A_d)
        else:
            loss = self.loss_fn(pred, y, self.adj, A_d) + torch.mean(g_loss)
        self.val_step_outputs.append(loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        X, X_d, y = batch
        X= X.permute(0,2,1)
        A_d = None
        if self.args['model_type'] != 'AttGRU':
            pred = self.model(X, self.adj, X_d) # model input_shape: x(B*seq*N)
        else:
            pred, A_d, _, _, g_loss = self.model(X, self.adj, X_d) # AttGRU output
        pred, y = self.format_output(pred, y)
        if self.args['model_type'] != 'AttGRU':
            loss = self.loss_fn(pred, y, self.adj, A_d)
        else:
            loss = self.loss_fn(pred, y, self.adj, A_d) + torch.mean(g_loss)
        return loss

    def configure_optimizers(self):
        if self.args['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['WEIGHT_DECAY'])
        else:
            raise ValueError('optimizer type error')
        return optimizer
    
    def _get_loss_fn(self):
        return MyLossFun(self.args['model_type'])
    
    def get_loss(self):
        return self._loss['training'], self._loss['validate'][1:]
    
    def set_gloabl_adj(self, adj):
        self.adj = adj
        
    def _free_step_outputs(self, phase):
        """Free up the memory of the step outputs."""
        if phase == 'training':
            self.training_step_outputs.clear()
        elif phase == 'validate':
            self.val_step_outputs.clear()


class MyLossFun(nn.Module):
    def __init__(self, model_name, lam=0.1) -> None:
        '''
        lam(float): 用于控制交叉熵的权重
        '''
        super().__init__()
        self.model_name = model_name
        self.lam = 0 #lam
        self.mse_loss = nn.MSELoss()
        self.crossentropy_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred, real, A_p, A_d):
        """
        :param _type_ pred: _description_
        :param _type_ real: _description_
        :param _type_ A_p: 实际邻接矩阵
        :param _type_ A_d: 预测动态邻接矩阵
        """
        if self.model_name != 'AttGRU':
            return self.mse_loss(pred, real)
        else:
            A_p = A_p.repeat(pred.shape[0], 1, 1) # (N*N) -> (B * N * N)
            return self.mse_loss(pred, real) + \
                self.lam * self.crossentropy_loss(A_d, A_p)