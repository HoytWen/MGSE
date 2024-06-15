import torch
import GCL.augmentors as A
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import dropout_adj
from typing import Optional

from torch.distributions.bernoulli import Bernoulli

def mask_node(x, edge_index, keep_prob, num_atom_type=119):
    num_nodes = edge_index.max().item() + 1
    probs = torch.tensor([keep_prob for _ in range(num_nodes)])
    dist = Bernoulli(probs)
    subset = dist.sample().to(torch.bool).to(edge_index.device)
    x[subset] = torch.tensor([num_atom_type, 0]).to(edge_index.device)

    return x

def drop_node(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, keep_prob: float = 0.5) -> (torch.Tensor, Optional[torch.Tensor]):
    num_nodes = edge_index.max().item() + 1
    probs = torch.tensor([keep_prob for _ in range(num_nodes)])
    dist = Bernoulli(probs)
    subset = dist.sample().to(torch.bool).to(edge_index.device)
    edge_index, edge_weight = subgraph(subset, edge_index, edge_weight)
    return edge_index, edge_weight

class AttributeMasking(Augmentor):
    def __init__(self, pf: float):
        super(AttributeMasking, self).__init__()
        self.pf = pf
    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = mask_node(x=x, edge_index=edge_index, keep_prob=self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe
    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

class NodeDropping(Augmentor):
    def __init__(self, pn: float):
        super(NodeDropping, self).__init__()
        self.pn = pn
    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = drop_node(edge_index, edge_weights, keep_prob=1. - self.pn)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)