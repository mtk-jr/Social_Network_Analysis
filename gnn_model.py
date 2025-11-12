import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

NUM_NODE_FEATURES = 4
NUM_CLASSES = 1
HIDDEN_CHANNELS = 32

class ContagionGAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(NUM_NODE_FEATURES, HIDDEN_CHANNELS, heads=4, edge_dim=1)
        self.conv_mid = GATConv(HIDDEN_CHANNELS*4, HIDDEN_CHANNELS, heads=4, edge_dim=1)
        self.conv2 = GATConv(HIDDEN_CHANNELS*4, NUM_CLASSES, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv_mid(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x

