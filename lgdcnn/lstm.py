import numpy as np
import torch
from torch import nn
from lgdcnn.composition import *
# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

class textrnn(nn.Module):
    def __init__(self, embed=512, hidden_dimension=256, num_layers=1,
        bidirectional=True):
        super(textrnn, self).__init__()
        
        self.embed = embed
        self.hidden_dimension= hidden_dimension
        self.bidirectional = bidirectional
        self.num_layers = num_layers 

        self.rnn = torch.nn.LSTM(
            self.embed, self.hidden_dimension,
            num_layers=self.num_layers, bias=True,
            batch_first=True, dropout=0, bidirectional=self.bidirectional,
            ) 

    def forward(self, x):
        output, _ = self.rnn(x)  
            
        return output


class BaseModel(nn.Module):
    def __init__(self, d_model, frac=False, compute_device=None,):
        super().__init__()
        self.d_model = d_model
        self.fractional = frac
        self.compute_device = compute_device

        self.embed = ElementEmbedder(embedding_size=self.d_model, compute_device=self.compute_device)
        self.pe = StoichiometryEmbedder(self.d_model, resolution_value=5000, use_log10=False, compute_device=self.compute_device)
        self.ple = StoichiometryEmbedder(self.d_model, resolution_value=5000, use_log10=True, compute_device=self.compute_device)
        self.emb_scaler = nn.Parameter(torch.tensor([1.0]))
        self.pos_scaler = nn.Parameter(torch.tensor([1.0]))
        self.pos_scaler_log = nn.Parameter(torch.tensor([1.0]))
        self.textrnn = textrnn()
        
    def forward(self, src, frac):
        x = self.embed(src) * 2 ** self.emb_scaler
        all_pe_matrix = self._calculate_pe(x, frac)
        x_src = x + all_pe_matrix
        x = self.textrnn(x_src)
        return x

    def _calculate_pe(self, x, frac):
        pe_matrix  = torch.zeros_like(x)
        ple_matrix  = torch.zeros_like(x)
        pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
        ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
        pe_matrix[:, :, :self.d_model // 2] = self.pe(frac) * pe_scaler
        ple_matrix[:, :, self.d_model // 2:] = self.ple(frac) * ple_scaler
        return pe_matrix + ple_matrix


class DiscreteModel(nn.Module):
    def __init__(self, out_dims=3, d_model=512, filter_map=512, compute_device=None, out=True):
        super().__init__()
        self.out = out
        self.out_dims = out_dims
        self.d_model = d_model
        self.filter_map = filter_map
        self.compute_device = compute_device
        self.encoder = BaseModel(d_model=self.d_model, compute_device=self.compute_device)
        self.output_nn = OutputNN(self.filter_map, self.out_dims)

    def forward(self, src, frac):
        output = self.encoder(src, frac)
        output = self._process_output(output, src)
        return output

    def _process_output(self, output, src):
        mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(output)
        if self.out:
            output = output.masked_fill(mask, 0)
            output = output.sum(dim=1) / (~mask).sum(dim=1)
            output, logits = output.chunk(2, dim=-1)
            probability = torch.ones_like(output)
            probability[:, :logits.shape[-1]] = torch.sigmoid(logits)
            output = output * probability
        return output
