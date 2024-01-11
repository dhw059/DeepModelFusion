import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

# To get the absolute path of the directory where the lgdcnn module is located
lgdcnn_dir = r"D:\deep\LGDCNN"

class ElementEmbedder(nn.Module):
    def __init__(self, embedding_size, compute_device=None, Emebedder_Choice=True):
        super(ElementEmbedder, self).__init__()
        self.embedding_size = embedding_size
        self.compute_device = compute_device
        element_dir = os.path.join(lgdcnn_dir, "data", "element_properties")
        if Emebedder_Choice:
            emb2vec_file = os.path.join(element_dir, "mat2vec.csv")
        else:   
            emb2vec_file = os.path.join(element_dir, "onehot.csv")
        element_properties = pd.read_csv(emb2vec_file, index_col=0).values
        feature_size = element_properties.shape[-1]

        self.cbfv = nn.Embedding.from_pretrained(
            torch.tensor(np.concatenate([np.zeros((1, feature_size)), element_properties]), dtype=torch.float32)
        ).to(self.compute_device, dtype=torch.float32)
        self.fc_mat2vec = nn.Linear(feature_size, embedding_size).to(self.compute_device)

    def forward(self, input):
        embedded_input = self.cbfv(input)
        transformed_input = self.fc_mat2vec(embedded_input)
        return transformed_input
    

class StoichiometryEmbedder(nn.Module):
    def __init__(self, d_model, resolution_value=100, use_log10=False, compute_device=None):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution_value = resolution_value
        self.use_log10 = use_log10
        self.compute_device = compute_device

        x = torch.linspace(0, resolution_value - 1, resolution_value, requires_grad=False).view(resolution_value, 1)
        fraction = torch.linspace(0, self.d_model - 1, self.d_model, requires_grad=False).view(1, self.d_model).repeat(resolution_value, 1)
        pe = torch.zeros(resolution_value, self.d_model)
        pe[:, 0::2] = self._calculate_pe_values(x, fraction[:, 0::2], self.d_model)
        pe[:, 1::2] = self._calculate_pe_values(x, fraction[:, 1::2], self.d_model)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = self._apply_log10_transformation(x)
        x = self._apply_clamp(x)
        frac_idx = self._calculate_frac_idx(x)
        out = self.pe[frac_idx.to(torch.long)]
        return out

    def _calculate_pe_values(self, x, fraction, d_model):
        return torch.sin(x / torch.pow(50, 2 * fraction / d_model))

    def _apply_log10_transformation(self, x):
        if self.use_log10:
            x = 0.0025 * (torch.log2(x)) ** 2
            x = torch.clamp(x, max=1)
        return x

    def _apply_clamp(self, x):
        x = torch.clamp(x, min=1 / self.resolution_value, max=1)
        return x

    def _calculate_frac_idx(self, x):
        frac_idx = torch.round(x * self.resolution_value) - 1
        return frac_idx


class OutputNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OutputNN, self).__init__()
        dims = [input_dim]
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        return self.fc_out(fea)

    def __repr__(self):
        return f'{self.__class__.__name__}'

  