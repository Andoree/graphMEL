import torch
from torch import nn
from torch.nn.modules.module import Module


class DistMult(Module):
    def __init__(self, num_rel, rel_emb_size):
        super(DistMult, self).__init__()

        # Create weights & biases
        self.num_rel = num_rel
        self.rel_emb_size = rel_emb_size
        self.relation_embeddings = nn.Embedding(num_rel, rel_emb_size)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)

    def forward(self, src_node_embs, trg_node_embs, rel_ids):
        """
        source_node_embeddings: <batch_size, emb_size>
        target_node_embeddings:  <batch_size, emb_size>
        rel_ids : <batch_size>
        """
        rel_embs = self.relation_embeddings(rel_ids)
        scores = (src_node_embs * rel_embs * trg_node_embs).sum(dim=-1)
        reg = torch.mean(src_node_embs ** 2) + torch.mean(rel_embs ** 2) + torch.mean(trg_node_embs ** 2)

        return scores, reg
