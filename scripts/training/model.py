import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEOverBert(nn.Module):
    def __init__(self, bert_encoder, hidden_channels, num_layers, graphsage_dropout, multigpu_flag):
        super(GraphSAGEOverBert, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.bert_encoder = bert_encoder
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        self.graphsage_dropout = graphsage_dropout

        for i in range(num_layers):
            in_channels = self.bert_hidden_dim if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, input_ids, attention_mask, adjs):
        last_hidden_states = self.bert_encoder(input_ids, attention_mask=attention_mask,
                                               return_dict=True)['last_hidden_state']
        x = torch.stack([elem[0, :] for elem in last_hidden_states])
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.graphsage_dropout, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.graphsage_dropout, training=self.training)
        return x
