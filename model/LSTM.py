import torch
import torch.nn as nn


class Model(nn.Module):
    """
     VanillaRNN is the most direct and traditional method for time series prediction using RNN-class methods.
     It completes multi-variable long time series prediction through multi-variable point-wise input and cyclic prediction.
     """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.rnn = nn.LSTM(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)

        self.predict = nn.Sequential(
            nn.Linear(self.d_model, self.enc_in)
        )

    def forward(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x = x_enc # b,s,c

        # encoding
        _, (hn, cn) = self.rnn(x)
        # decoding
        y = []
        for i in range(self.pred_len):
            yy = self.predict(hn)  # 1,b,c
            yy = yy.permute(1, 0, 2)  # b,1,c
            y.append(yy)
            _, (hn, cn) = self.rnn(yy, (hn, cn))
        dec_out = torch.stack(y, dim=1).squeeze(2) # bc,s,1

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out