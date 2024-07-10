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
    def __init__(self, embedding_size, compute_device=None):
        super(ElementEmbedder, self).__init__()
        self.embedding_size = embedding_size
        self.compute_device = compute_device

        element_dir = os.path.join(lgdcnn_dir, "data", "element_properties")
        mat2vec_file = os.path.join(element_dir, "mat2vec.csv")
        element_properties = pd.read_csv(mat2vec_file, index_col=0).values
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

  
class DPCNNModify(nn.Module):
    def __init__(self, num_filters=512, embed=512, kernel_size=3, blocks=2 ):
        super(DPCNNModify, self).__init__()
        self.num_filters = num_filters
        self.embed = embed
        self.kernel_size = kernel_size
        self.radius =  1
        self.blocks = blocks
        self.pooling_stride = 1 
        self.pool_kernel_size = 1
        
        self.convert_conv = torch.nn.Sequential(
                torch.nn.Conv1d(
                self.embed, self.num_filters,         
                self.kernel_size, padding=self.radius))

        self.convs = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_filters, self.num_filters,  
                self.kernel_size, padding=self.radius),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_filters, self.num_filters,
                self.kernel_size, padding=self.radius)
        ) for _ in range(self.blocks + 1)])  

    def forward(self, x):
        embedding = x.permute(0, 2, 1)
        conv_embedding = self.convert_conv(embedding) 
        conv_features = self.convs[0](conv_embedding)
        conv_features = conv_embedding + conv_features  
        for i in range(1, len(self.convs)):
            block_features = F.max_pool1d(
                conv_features, self.pool_kernel_size, self.pooling_stride)
            conv_features = self.convs[i](block_features)
            conv_features = conv_features + block_features
        return conv_features.permute(0,2,1)


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
        self.dpcnnmodify = DPCNNModify()
        
    def forward(self, src, frac):
        x = self.embed(src) * 2 ** self.emb_scaler
        all_pe_matrix = self._calculate_pe(x, frac)
        x_src = x + all_pe_matrix
        x = self.dpcnnmodify(x_src)
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
