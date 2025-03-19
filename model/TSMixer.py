import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, 4*configs.d_model),
            nn.ReLU(),
            nn.Linear(4*configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, 4*configs.d_model),
            nn.ReLU(),
            nn.Linear(4*configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.layer = configs.layers
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.layers)])
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        dec_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len, 1))
        return dec_out
