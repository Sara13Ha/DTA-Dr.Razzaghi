import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import math
from torch_scatter import scatter_softmax

#YASSI CHECK (Cross Attention)
#SARA CHECK

class DrugProteinCrossAttention(nn.Module):
    def __init__(self, d_drug_k, d_protein_n, d_head=128):
        super(DrugProteinCrossAttention, self).__init__()
        self.q_proj = nn.Linear(d_drug_k, d_head)
        self.k_proj = nn.Linear(d_protein_n, d_head)
        self.v_proj = nn.Linear(d_protein_n, d_drug_k) 
        self.sqrt_d = math.sqrt(d_head)

    def forward(self, drug_nodes, protein_vec, drug_batch_index):
        q = self.q_proj(drug_nodes)
        k = self.k_proj(protein_vec)
        v = self.v_proj(protein_vec)
        k_expanded = k[drug_batch_index]
        v_expanded = v[drug_batch_index]
        scores = (q * k_expanded).sum(dim=1, keepdim=True) / self.sqrt_d
        attn_weights = scatter_softmax(scores, drug_batch_index, dim=0)
        context = attn_weights * v_expanded
        output = drug_nodes + context
        return output


class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, output_dim=128, dropout=0.2):
        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        self.protein_n_features = 256

        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.fc_g1 = nn.Linear(num_features_xd * 10, self.output_dim) 
        self.drug_pool = gmp 
        self.protein_fc = nn.Linear(self.protein_n_features, self.protein_n_features) 
        self.attention = DrugProteinCrossAttention(
            d_drug_k=self.output_dim,           # k=128
            d_protein_n=self.protein_n_features # n=256
        )

        final_dim = self.output_dim + self.protein_n_features # 128 + 256 = 384
        self.fc1 = nn.Linear(final_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
    
        x, edge_index, batch = data.x, data.edge_index, data.batch
        protein_vec = data.protein_vec # [batch, 1, n]
        protein_vec = protein_vec.squeeze(1) # [batch, n]
        protein_vec = self.relu(self.protein_fc(protein_vec))
        protein_vec = self.dropout(protein_vec)
    
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        drug_nodes = self.relu(x) 
        drug_nodes_projected = self.relu(self.fc_g1(drug_nodes)) # [total_atoms, k=128] (Query)
        attended_drug_nodes = self.attention(drug_nodes_projected, protein_vec, batch) # [total_atoms, k=128]
        final_drug_vec = self.drug_pool(attended_drug_nodes, batch) # [batch, k=128]
        final_drug_vec = self.dropout(final_drug_vec)
        xc = torch.cat((final_drug_vec, protein_vec), 1) # [batch, k+n = 128+256=384]

        x = self.fc1(xc)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out