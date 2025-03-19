import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.pred_len = configs.pred_len
        self.label_len=configs.seq_len//2
        # Embedding
        self.enc_embedding = nn.Linear(configs.enc_in, configs.d_model)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_model,
                    dropout=configs.dropout,
                    activation='gelu'
                ) for l in range(configs.layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.dec_embedding = nn.Linear(configs.enc_in, configs.d_model)
        self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, 1, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, 1, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_model,
                        dropout=configs.dropout,
                        activation='gelu',
                    )
                    for l in range(1)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.enc_in, bias=True)
            )

    def forward(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        B,T,N=x_enc.shape
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        zeros = torch.zeros([B, self.pred_len,
                             N], device=x_enc.device)
        x_dec = torch.cat(
            [x_enc[:, -self.label_len:, :], zeros], dim=1)
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out)[:,-self.pred_len:,:]
        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out