import torch
import torch.nn as nn
from layers.Leddam_Layer import Leddam

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.pred_len=configs.pred_len
        self.leddam=Leddam(configs.enc_in,configs.seq_len,configs.d_model,
                       configs.dropout,pe_type='zeros',kernel_size=5,n_layers=configs.layers)
        self.Linear_main = nn.Linear(configs.d_model, configs.pred_len) 
        self.Linear_res = nn.Linear(configs.d_model, configs.pred_len)
        self.Linear_main.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        self.Linear_res.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
    def forward(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding
        res,main=self.leddam(x_enc)
        main_out=self.Linear_main(main.permute(0,2,1)).permute(0,2,1)
        res_out=self.Linear_res(res.permute(0,2,1)).permute(0,2,1)
        dec_out=main_out+res_out
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out