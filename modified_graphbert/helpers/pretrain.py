import numpy as np
import torch
import os

from graph_bert.MethodBertComp import GraphBertConfig
from graph_bert.MethodGraphBertNodeConstruct import MethodGraphBertNodeConstruct
from graph_bert.MethodGraphBertGraphRecovery import MethodGraphBertGraphRecovery
from helpers.preprocess import AbstractModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(1)
torch.manual_seed(1)


class GraphBertNodeAttributeReconstruction(AbstractModel):
    lr = 0.001
    k = 7
    max_epoch = 200
    residual_type = 'graph_raw'
    nfeature = x_size = 1433
    y_size = 7
    graph_size = 2708
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2

    def __init__(self, data_obj, dataset_name, pretrained_path, max_index=116):
        super().__init__(data_obj, dataset_name)
        self.data_obj = data_obj
        self.dataset_name = dataset_name
        self.pretrained_path = pretrained_path
        self.bert_config = GraphBertConfig(
            residual_type=self.residual_type, k=self.k,
            x_size=self.nfeature, y_size=self.y_size, hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size, num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers, max_wl_role_index=max_index,
            max_hop_dis_index=max_index, max_inti_pos_index=max_index)

    def graphbert_reconstruct(self, device='cpu'):
        method_obj = MethodGraphBertNodeConstruct(self.bert_config, device=device)
        method_obj.max_epoch = self.max_epoch
        method_obj.lr = self.lr
        method_obj.save_pretrained_path = self.pretrained_path

        self.run(method_obj, './result/GraphBert/')

    def graphbert_network_recovery(self, device='cpu'):
        method_obj = MethodGraphBertGraphRecovery(self.bert_config, self.pretrained_path, device=device)
        method_obj.max_epoch = self.max_epoch
        method_obj.lr = self.lr
        method_obj.save_pretrained_path = self.pretrained_path

        self.run(method_obj, './result/GraphBert/')
